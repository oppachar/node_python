const express = require('express')
const router = express.Router();


// config.js에서 export한 모듈을 다음과 같이 import 시킬 수 있다. (같은 디렉토리 위치)
const database = require('./config');

// localhost:3000/firebase/save 호출
router.get('/save', function(req, res){
    database.ref('customer').set({name : "junseok"}, function(error) {
        if(error)
            console.error(error)
        else
            console.log("success save !!");
    });
    return res.json({firebase : true});
});


module.exports = router;