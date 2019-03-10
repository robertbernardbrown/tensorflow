import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-node";
import iris from "./iris.json";
import irisTesting from "./iris-testing.json";

const trainingData = tf.tensor2d(iris.map(item => [
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]))
const outputData = tf.tensor2d(iris.map(item => [
    item.species === "setosa" ? 1 : 0,
    item.species === "virginica" ? 1 : 0,
    item.species === "versicolor" ? 1 : 0,
]))
const testingData = tf.tensor2d(iris.map(item => [
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]))