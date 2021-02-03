library(tidyverse)
library(RColorBrewer)
library(knitr)
library(kableExtra)
library(MASS)
library(caret)
library(randomForest)
library(xgboost)

###
# Intro
###

# Se cargan los datos
table_1 <- read_csv("data/student-mat.csv", col_types = "ffifffffffffiinffffffffffffffnnnn")
table_2 <- read_csv("data/student-por.csv", col_types = "ffifffffffffiinffffffffffffffnnnn")

# Se juntan las tablas
data <- full_join(table_1, table_2, by = c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet","guardian","traveltime","studytime","schoolsup","famsup","activities","higher","romantic","famrel","freetime","goout","Dalc","Walc","health"))

# Se colapsan las variables específicas de cada asignatura y se arreglan las demás variables
data <- data %>%
  mutate(failures = rowMeans(cbind(failures.x, failures.y), na.rm = TRUE),
         paid = as.factor(ifelse(paid.x == "yes" | paid.y == "yes", "yes", "no")),
         absences = rowMeans(cbind(absences.x, absences.y), na.rm = TRUE),
         G1 = rowMeans(cbind(G1.x, G1.y), na.rm = TRUE),
         G2 = rowMeans(cbind(G2.x, G2.y), na.rm = TRUE),
         G3 = rowMeans(cbind(G3.x, G3.y), na.rm = TRUE)) %>%
  dplyr::select(-matches("failures."), -matches("paid."), -matches("absences."), -matches("G1."), -matches("G2."), -matches("G3."), -school, -Dalc) %>%
  mutate(paid = as.factor(ifelse(is.na(paid) | paid == "no", "no", "yes")),
         sex = factor(sex, labels = c("Mujer", "Hombre")),
         famsize = ordered(famsize, levels = c("LE3","GT3")),
         Medu = ordered(Medu, levels = c(0,1,2,3,4)),
         Fedu = ordered(Fedu, levels = c(0,1,2,3,4)),
         famrel = ordered(famrel, levels = c(1,2,3,4,5)),
         freetime = ordered(freetime, levels = c(1,2,3,4,5)),
         goout = ordered(goout, levels = c(1,2,3,4,5), labels = c("Nada","Poco","Algo","Bastante","Mucho")),
         Walc = ordered(Walc, levels = c(1,2,3,4,5), labels = c("Nada","Poco","Algo","Bastante","Mucho")),
         health = ordered(health, levels = c(1,2,3,4,5)))


###
# EDA
###

ggplot(data) +
  geom_bar(aes(x = Walc, fill = Walc), color = "black", show.legend = FALSE) +
  ggtitle("Distribución de los estudiantes por consumo de alcohol") +
  xlab("Consumo de alcohol los fines de semana") +
  ylab("Frecuencia") +
  scale_fill_manual(values = brewer.pal(n = 5, name = "OrRd")) +
  geom_hline(yintercept = 0, color = "black") +
  geom_text(aes(label = paste(as.character(round(length(which(Walc == "Nada")) / nrow(data) * 100,1)),"%", sep = "")), x = "Nada", y = length(which(data$Walc == "Nada")) / 2, check_overlap = TRUE) +
  geom_text(aes(label = paste(as.character(round(length(which(Walc == "Poco")) / nrow(data) * 100,1)),"%", sep = "")), x = "Poco", y = length(which(data$Walc == "Poco")) / 2, check_overlap = TRUE) +
  geom_text(aes(label = paste(as.character(round(length(which(Walc == "Algo")) / nrow(data) * 100,1)),"%", sep = "")), x = "Algo", y = length(which(data$Walc == "Algo")) / 2, check_overlap = TRUE) +
  geom_text(aes(label = paste(as.character(round(length(which(Walc == "Bastante")) / nrow(data) * 100,1)),"%", sep = "")), x = "Bastante", y = length(which(data$Walc == "Bastante")) / 2, check_overlap = TRUE) +
  geom_text(aes(label = paste(as.character(round(length(which(Walc == "Mucho")) / nrow(data) * 100,1)),"%", sep = "")), x = "Mucho", y = length(which(data$Walc == "Mucho")) / 2, check_overlap = TRUE)

ggplot(data) +
  geom_bar(aes(x = sex, fill = Walc), color = "black", position = "fill") +
  ggtitle("Consumo de alcohol por sexo") +
  xlab("Sexo") +
  ylab("Proporción") +
  scale_fill_manual(values = brewer.pal(n = 5, name = "OrRd")) +
  geom_hline(yintercept = 0, color = "black")

ggplot(data) +
  geom_jitter(aes(x = age, y = G3, fill = Walc), shape = 21, color = "black") +
  ggtitle("Consumo de alcohol por edad y calificaciones", "Jitter Plot") +
  xlab("Edad") +
  ylab("Calificaciones finales") +
  scale_fill_manual(values = brewer.pal(n = 5, name = "OrRd")) +
  ylim(-0.2,20)

ggplot(data) +
  geom_bar(aes(x = goout, fill = Walc), color = "black") +
  ggtitle("Consumo de alcohol por costumbre de salir con amigos") +
  xlab("Sale con los amigos") +
  ylab("Frecuencia") +
  scale_fill_manual(values = brewer.pal(n = 5, name = "OrRd")) +
  geom_hline(yintercept = 0, color = "black")


###
# LDA
###

# Estimación del modelo con Leave-One-Out Cross-Validation
fit_lda <- lda(Walc ~ ., data = data, CV = TRUE)
lda_acc <- mean(fit_lda$class == as.character(data$Walc))

kable(t(table(fit_lda$class, data$Walc)), align = c("c","c","c","c","c")) %>%
  kable_styling() %>%
  add_header_above(c(" " = 1, "Walc Predicho" = 5))


###
# Random Forest
###

# Se dividen los datos entre train y test
set.seed(1111)
train_i <- createDataPartition(data$Walc, p = 0.8, list = FALSE)
train_data <- data[train_i,]
test_data <- data[-train_i,]
# Se crean las divisiones para hacer 5-Fold Cross-Validation
train_out <- createFolds(train_data$Walc, k = 5, returnTrain = TRUE)
test_out <- lapply(train_out, function (x) {(1:nrow(train_data))[-x]})
# Se eligen los diferentes valores que se quieren probar
rf_grid <- expand.grid(mtry = c(2,3,4,6,9,12))
# Se crea la parrilla para ver el mejor valor de cada caja
rf_tune <- matrix(nrow = 5, ncol = 1)
dimnames(rf_tune) = list(c("k1", "k2", "k3", "k4", "k5"), c("mtry"))
# Se crea una variable para guardar la precisión del modelo
rf_acc <- 0
# Se especifican los parámetros que se usarán para estimar los modelos
fitControl <- trainControl(method = "cv", number = 5)
# Se hace el 5-Fold Cross-Validation
for (i in 1:5) {
  # Se elige la caja
  train_in <- data[train_out[[i]],]
  test_in <- data[test_out[[i]],]
  # Se estima el modelo de cada caja
  fit_rf <- train(Walc ~ .,
                  data = train_in,
                  method = "rf",
                  trControl = fitControl,
                  tuneGrid = rf_grid,
                  metric = "Accuracy",
                  distribution = "multinomial")
  # Se actualiza la variable que guarda el hiperparámetro óptimo
  rf_tune[i,1] <- as.matrix(fit_rf$bestTune)
}

kable(rf_tune, align = c("c","c")) %>%
  kable_styling()
# Se estima el modelo con el hiperparámetro óptimo y los datos train
fit_rf <- randomForest(Walc ~ ., data = train_data, mtry = 9, ntree = 1000)
# Se predicen los datos test
pred_rf <- predict(fit_rf, test_data)
# Se compara la predicción con el valor real
rf_acc <- mean(as.character(pred_rf) == as.character(test_data$Walc))


###
# XGB
###

# Se dividen los datos entre train y test
set.seed(1111)
train_i <- createDataPartition(data$Walc, p = 0.8, list = FALSE)
train_data <- data[train_i,]
test_data <- data[-train_i,]
# Se crean las divisiones para hacer 5-Fold Cross-Validation
train_out <- createFolds(train_data$Walc, k = 5, returnTrain = TRUE)
test_out <- lapply(train_out, function (x) {(1:nrow(train_data))[-x]})
# Se eligen los diferentes valores que se quieren probar
xgb_grid <- expand.grid(nrounds=c(50),
                        max_depth = c(3, 6, 10),
                        eta = c(0.01, 0.1, 0.3),
                        gamma = c(0.01),
                        colsample_bytree = c(0.5, 1),
                        min_child_weight = c(0, 1),
                        subsample = c(0.5, 1))
# Se crea la parrilla para ver los mejores valores de cada caja
xgb_tune <- matrix(nrow = 5, ncol = 7)
dimnames(xgb_tune) = list(c("k1", "k2", "k3", "k4", "k5"), c("nrounds", "max_depth", "eta", "gamma", "colsample_bytree", "min_child_weight", "subsample"))
# Se crea una variable para guardar la precisión del modelo
xgb_acc <- 0
# Se especifican los parámetros que se usarán para estimar los modelos
fitControl <- trainControl(method = "cv", number = 5)
# Se hace el 5-Fold Cross-Validation
for (i in 1:5) {
  # Se elige la caja
  train_in <- data[train_out[[i]],]
  test_in <- data[test_out[[i]],]
  # Se estima el modelo de cada caja
  fit_xgb <- train(Walc ~ ., data = train_in, method = "xgbTree", trControl = fitControl, tuneGrid = xgb_grid, metric = "Accuracy")
  # Se actualiza la variable que guarda el hiperparámetro óptimo
  xgb_tune[i,] <- as.matrix(fit_xgb$bestTune)
}

kable(xgb_tune, align = c("c","c","c","c","c","c","c","c")) %>%
  kable_styling()
# Se estima el modelo con los hiperparámetros óptimos y los datos train
xgb_bst_grid <- expand.grid(nrounds=c(50),
                            max_depth = c(3),
                            eta = c(0.01),
                            gamma = c(0.01),
                            colsample_bytree = c(1),
                            min_child_weight = c(1),
                            subsample = c(0.5))
fit_xgb <- train(Walc ~ ., data = train_data, method = "xgbTree", tuneGrid = xgb_bst_grid, metric = "Accuracy")
# Se predicen los datos test
pred_xgb <- predict(fit_xgb, test_data)
# Se compara la predicción con el valor real
xgb_acc <- mean(as.character(pred_xgb) == as.character(test_data$Walc))

