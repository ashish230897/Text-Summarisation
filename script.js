function predict(){

  let val = document.getElementById("myText").value;
  console.log(val);
  let data = {"text": val};

  fetch("http://127.0.0.1:12345/generate", {
  method: "POST",
  mode: 'cors',
  headers: {'Content-Type': 'application/json'}, 
  body: JSON.stringify(data)
  // }).then(res => console.log(res.json()));
  }).then(res => res.json()).then(
    (myBlob) => {
      console.log("result is ", myBlob["summary"]);
      
      const currentDiv = document.getElementById("result");
      currentDiv.innerHTML = '';

      var newDiv = document.createElement('div');
      var j = 0;
      newDiv.innerHTML += "summary: " + myBlob["summary"] + "<br><br>";
      newDiv.innerHTML += "Input saliency map is" + " : ";
      
      for(j = 0; j < myBlob["attention"].length; j++){
        newDiv.innerHTML += myBlob["attention"][j] + " ";
      }
      
      currentDiv.appendChild(newDiv);
      


    }
  );
}