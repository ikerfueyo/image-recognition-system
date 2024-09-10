document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const resultDiv = document.getElementById('result');
    const labelSpan = document.getElementById('label').querySelector('span');
    const confidenceSpan = document.getElementById('confidence').querySelector('span');

    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        const formData = new FormData();
        const imageInput = document.getElementById('image');
        formData.append('image', imageInput.files[0]);

        fetch('/classify/', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            labelSpan.textContent = data.predicted_label;
            confidenceSpan.textContent = data.confidence;
            resultDiv.classList.remove('hidden');
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
});
