const checkbox = document.getElementById('check');
const getStarted = document.getElementById('summary-button');

checkbox.addEventListener('change', function() {
  if (checkbox.checked) {
    console.log('Checkbox is checked!');
	getStarted.classList.remove('gray');
	getStarted.disabled = false;
  } else {
    console.log('Checkbox is not checked.');
	getStarted.classList.add('gray');
	getStarted.disabled = true;
  }
});

function redirectToPage() {
	if (checkbox.checked) {
		window.open("static/main.html", '_blank');
	  } 
}
