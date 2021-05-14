// Firebase App (the core Firebase SDK) is always required and
// must be listed before other Firebase SDKs
const firebase = require("firebase");

const firebaseConfig = {
    apiKey: "AIzaSyDux1gkR4W7-l2JHQDSthSVR-0fcHQARzQ",
    authDomain: "persona-d1ed9.firebaseapp.com",
    projectId: "persona-d1ed9",
    storageBucket: "persona-d1ed9.appspot.com",
    messagingSenderId: "260737152566",
    appId: "1:260737152566:web:2aa6d62b58094848554a97",
    measurementId: "G-X82QKZMXL9"
};
firebase.initializeApp(firebaseConfig)
let database = firebase.database();

module.exports = database;