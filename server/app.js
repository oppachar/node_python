const express = require('express')
const app = express()
const port = 8000
const firebaseRouter = require('./routers/firebase/service');
const { spawn } = require('child_process')


app.get('/', (req, res) => {
  //res.send('Hello World!')

  let dataToSend
  let largeDataSet = []

  const python = spawn('python', ['face_recongition/features.py'])

  python.stdout.on('data', function (data){
    console.log('Pipe data from python script ...')

    largeDataSet.push(data)
  })

  python.on('close', (code) => {
    console.log(`child process close all stdio with code ${code}`)

    res.send(largeDataSet.join(''))
  })
})
app.use('/firebase',firebaseRouter);


app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`)
})