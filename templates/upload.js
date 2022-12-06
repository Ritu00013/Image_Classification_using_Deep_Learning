const imageupload = document.getElementById('image_upload');

const fileChosen = document.getElementById('file-chosen');

imageupload.addEventListener('change', function(){
  fileChosen.textContent = this.files[0].name
})
