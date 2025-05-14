document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const imageFileInput = document.getElementById('imageFile');
    const noFileMessage = document.getElementById('no-file-message');
    const uploadedImage = document.getElementById('uploadedImage');
    const colorizedImage = document.getElementById('colorizedImage');
    const downloadLink = document.getElementById('downloadLink');
    const errorMessage = document.getElementById('error-message');
    const loadingSpinner = document.getElementById('loadingSpinner');
    let uploadedImageURL = null;

    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault();

        if (!imageFileInput.files.length) {
            noFileMessage.textContent = 'Please select an image file';
            return;
        }

        noFileMessage.textContent = '';
        errorMessage.textContent = '';
        loadingSpinner.style.display = 'block'; // Show loading spinner

        const formData = new FormData();
        formData.append('file', imageFileInput.files[0]);

        fetch('/upload', {  // Ensure we are calling the correct endpoint
            method: 'POST',
            body: formData
        })
        .then(response => {
            loadingSpinner.style.display = 'none'; // Hide loading spinner
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.blob(); // Get response as blob
        })
        .then(imageBlob => {
            // Revoke the previous object URL to prevent memory leaks
            if (uploadedImageURL) {
                URL.revokeObjectURL(uploadedImageURL);
            }

            // Display the uploaded image
            uploadedImageURL = URL.createObjectURL(imageFileInput.files[0]);
            uploadedImage.src = uploadedImageURL;
            uploadedImage.style.display = 'block';

            // Create a URL for the colorized image
            const colorizedImageUrl = URL.createObjectURL(imageBlob);
            colorizedImage.src = colorizedImageUrl;
            colorizedImage.style.display = 'block';

            downloadLink.href = colorizedImageUrl; // Set download link
            downloadLink.style.display = 'block'; // Show download link
        })
        .catch(error => {
            loadingSpinner.style.display = 'none'; // Hide loading spinner
            errorMessage.textContent = 'Error: ' + error.message; // Display error message
        });
    });
});