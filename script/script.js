// create a global array to store the added symptoms
var addedSymptoms = [];

const symptomsDict = {'Itching': 0,
'Skin Rash': 1,
'Nodal Skin Eruptions': 2,
'Continuous Sneezing': 3,
'Shivering': 4,
'Chills': 5,
'Joint Pain': 6,
'Stomach Pain': 7,
'Acidity': 8,
'Ulcers On Tongue': 9,
'Muscle Wasting': 10,
'Vomiting': 11,
'Burning Micturition': 12,
'Spotting  urination': 13,
'Fatigue': 14,
'Weight Gain': 15,
'Anxiety': 16,
'Cold Hands And Feets': 17,
'Mood Swings': 18,
'Weight Loss': 19,
'Restlessness': 20,
'Lethargy': 21,
'Patches In Throat': 22,
'Irregular Sugar Level': 23,
'Cough': 24,
'High Fever': 25,
'Sunken Eyes': 26,
'Breathlessness': 27,
'Sweating': 28,
'Dehydration': 29,
'Indigestion': 30,
'Headache': 31,
'Yellowish Skin': 32,
'Dark Urine': 33,
'Nausea': 34,
'Loss Of Appetite': 35,
'Pain Behind The Eyes': 36,
'Back Pain': 37,
'Constipation': 38,
'Abdominal Pain': 39,
'Diarrhoea': 40,
'Mild Fever': 41,
'Yellow Urine': 42,
'Yellowing Of Eyes': 43,
'Acute Liver Failure': 44,
'Fluid Overload': 45,
'Swelling Of Stomach': 46,
'Swelled Lymph Nodes': 47,
'Malaise': 48,
'Blurred And Distorted Vision': 49,
'Phlegm': 50,
'Throat Irritation': 51,
'Redness Of Eyes': 52,
'Sinus Pressure': 53,
'Runny Nose': 54,
'Congestion': 55,
'Chest Pain': 56,
'Weakness In Limbs': 57,
'Fast Heart Rate': 58,
'Pain During Bowel Movements': 59,
'Pain In Anal Region': 60,
'Bloody Stool': 61,
'Irritation In Anus': 62,
'Neck Pain': 63,
'Dizziness': 64,
'Cramps': 65,
'Bruising': 66,
'Obesity': 67,
'Swollen Legs': 68,
'Swollen Blood Vessels': 69,
'Puffy Face And Eyes': 70,
'Enlarged Thyroid': 71,
'Brittle Nails': 72,
'Swollen Extremeties': 73,
'Excessive Hunger': 74,
'Extra Marital Contacts': 75,
'Drying And Tingling Lips': 76,
'Slurred Speech': 77,
'Knee Pain': 78,
'Hip Joint Pain': 79,
'Muscle Weakness': 80,
'Stiff Neck': 81,
'Swelling Joints': 82,
'Movement Stiffness': 83,
'Spinning Movements': 84,
'Loss Of Balance': 85,
'Unsteadiness': 86,
'Weakness Of One Body Side': 87,
'Loss Of Smell': 88,
'Bladder Discomfort': 89,
'Foul Smell Of urine': 90,
'Continuous Feel Of Urine': 91,
'Passage Of Gases': 92,
'Internal Itching': 93,
'Toxic Look (typhos)': 94,
'Depression': 95,
'Irritability': 96,
'Muscle Pain': 97,
'Altered Sensorium': 98,
'Red Spots Over Body': 99,
'Belly Pain': 100,
'Abnormal Menstruation': 101,
'Dischromic  Patches': 102,
'Watering From Eyes': 103,
'Increased Appetite': 104,
'Polyuria': 105,
'Family History': 106,
'Mucoid Sputum': 107,
'Rusty Sputum': 108,
'Lack Of Concentration': 109,
'Visual Disturbances': 110,
'Receiving Blood Transfusion': 111,
'Receiving Unsterile Injections': 112,
'Coma': 113,
'Stomach Bleeding': 114,
'Distention Of Abdomen': 115,
'History Of Alcohol Consumption': 116,
'Fluid Overload.1': 117,
'Blood In Sputum': 118,
'Prominent Veins On Calf': 119,
'Palpitations': 120,
'Painful Walking': 121,
'Pus Filled Pimples': 122,
'Blackheads': 123,
'Scurring': 124,
'Skin Peeling': 125,
'Silver Like Dusting': 126,
'Small Dents In Nails': 127,
'Inflammatory Nails': 128,
'Blister': 129,
'Red Sore Around Nose': 130,
'Yellow Crust Ooze': 131};

const allSymptoms = Object.keys(symptomsDict);

const symptomList = document.getElementById('symptom');


for (let i = 0; i < allSymptoms.length; i++) {
  const option = document.createElement('option');
  option.value = allSymptoms[i];
  option.text = allSymptoms[i];
  symptomList.appendChild(option);
}

function reRender(){
    while (symptomList.firstChild) {
        symptomList.removeChild(symptomList.firstChild);  
    }   
    for (let i = 0; i < allSymptoms.length; i++) {
        const option = document.createElement('option');
        option.value = allSymptoms[i];
        option.text = allSymptoms[i];
        symptomList.appendChild(option);
    }
}

const addBtn = document.getElementById("add");


function addSymptom() {
  // get the input and datalist elements
  var input = document.getElementById("symptom");
  // get the input value
  var symptom = input.value;
  
  // check if the input value is not empty and is not already in the added symptoms array
  if (symptom && !addedSymptoms.includes(symptom) && symptom != '--select symptom--') {
    // push it to the added symptoms array
    addedSymptoms.push(symptom);
    var myList  = document.getElementById("myList"); 
    const newListItem = document.createElement('li');
    newListItem.innerText = symptom
    myList.appendChild(newListItem);
    rememberIndex = allSymptoms.indexOf(symptom)
    allSymptoms.splice(rememberIndex, 1)
    reRender()
  }
  
  // clear the input value
  input.value = "";
}

const btnRemove = document.querySelector('#rem');
btnRemove.onclick = (e) => {
  e.preventDefault();

  //Removing the last Element
  var list = document.getElementById("myList");
  last = list.lastElementChild.innerText
  allSymptoms.push(last)
  reRender()
  list.removeChild(list.lastElementChild);
  var input = document.getElementById("symptom");
  input.value = "";
  addedSymptoms.pop()
}

function clearAll() {
    var list = document.getElementById("myList");
    while (list.firstChild) {
        list.removeChild(list.firstChild);   
    }
    while (symptomList.firstChild) {
        symptomList.removeChild(symptomList.firstChild);  
    }

    newAllSymptoms = Object.keys(symptomsDict);
    
    for (let i = 0; i < newAllSymptoms.length; i++) {
        const option = document.createElement('option');
        option.value = newAllSymptoms[i];
        option.text = newAllSymptoms[i];
        symptomList.appendChild(option);
    }
	document.getElementById("id1").textContent = '';
	document.getElementById("id2").textContent = '';
	document.getElementById("id3").textContent = '';
    var input = document.getElementById("symptom");
    input.value = "";
	addedSymptoms = []; // clear the array
}
function showSummary() {
    var xhr = new XMLHttpRequest();
    var url = 'http://localhost:8080/';
    xhr.open('POST', url, true);
    xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
    xhr.onreadystatechange = function() {
      if (xhr.readyState === 4 && xhr.status === 200) {
        var prediction = JSON.parse(xhr.responseText); // Parse JSON string into array
        console.log(prediction);
        document.getElementById("id1").textContent = prediction[0];
        document.getElementById("id2").textContent = prediction[1];
        document.getElementById("id3").textContent = prediction[2];
      }
    };
    var data = 'addedSymptoms=' + encodeURIComponent(addedSymptoms.join(','));
    xhr.send(data);
  
    // addedSymptoms = []; // clear the array

	// show alert after 10 seconds
	setTimeout(function() {
		var surveyUrl = 'https://forms.gle/1xUVyQ8WtYy8FzFS7';
		var message = 'Thank you for using our service. Please take a moment to complete our survey.';
		var link = document.createElement('a');
		link.textContent = 'Survey Link';
		link.href = surveyUrl;
		link.target = '_blank';
		var container = document.createElement('div');
		container.appendChild(document.createTextNode(message));
		// container.appendChild(link);
		var response = window.confirm(container.innerHTML + "\n\nWould you like to take the survey now?");
		if (response) {
			window.open(surveyUrl, '_blank');
		  }
		}, 3000);

	
	// var surveyUrl = 'https://forms.gle/1xUVyQ8WtYy8FzFS7';
    // window.open(surveyUrl, '_blank');
}
addBtn.addEventListener("click", addSymptom);