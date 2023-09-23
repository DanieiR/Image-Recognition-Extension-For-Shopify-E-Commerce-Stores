const express = require('express');
const bodyParser = require('body-parser');
const request = require('request');
const { PythonShell } = require('python-shell');
const axios = require('axios');
const app = express();

let shopifyoptions = {
  'method': 'GET',
  'url': 'https://75390d7a605fa3053ee913d87b0f1471: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@Novapixels.myshopify.com/admin/api/2023-01/products.json',
  'headers': {
    'Content-Type': 'application/json'
  }
}

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }));

app.use(function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  next();
});

app.post('/add', (req, res) => {
  let shopifyapidata = "";
  let pythondata = ""
  let myData = [];
   const inputimage = req.body;
axios(shopifyoptions)
  .then(function (response) {
    const responseBody = response.data; // Get the response body from the axios response object
   // console.log(responseBody);
   //  res.send(responseBody)
   // res.send(responseBody);
    shopifyapidata = responseBody
   // console.log(responseBody.products[0].id)
  })
  .catch(function (error) {
    console.error(error);
    res.status(500).send('Internal Server Error');
  });



  const options = {
    mode: 'text',
    pythonPath: 'C:/Users/moina/AppData/Local/Programs/Python/Python311/python.exe',
    pythonOptions: ['-u'],
    scriptPath: './NewProj',
    args: [JSON.stringify(inputimage)]
    
  };

  const pyshell = new PythonShell('testdata.py', options);
  let output = '';

  pyshell.send('Hello, Python!');

  pyshell.on('message', (message) => {
    pythondata = message;
     myData.push(message); // append each message to output
    console.log(message);
  });

  pyshell.on('error', (err) => {
    console.error('Python error:', err);
  });

  pyshell.end((err, code, signal) => {
    if (err) {
      console.error('Python script execution failed:', err);
      res.status(500).send('Internal Server Error');
    } else {
      console.log('Python script execution completed with code', code);
      res.send({ "shopifyData": shopifyapidata, "pythonData": myData });
      //res.send({"pythonData": myData });
    }
  });
});

app.get('/display', (req, res) => {

});

app.listen(3000, () => {
  console.log("Server running at port 3000")
});
