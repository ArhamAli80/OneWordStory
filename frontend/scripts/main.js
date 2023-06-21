// Get the elements
const wordInput = document.getElementById('word-input');
const submitBtn = document.getElementById('submit-btn');
const storyOutput = document.getElementById('story-output');

// Event listener for the submit button
submitBtn.addEventListener('click', generateStory);

wordInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      generateStory();
    }
  });

// Function to generate the story
function generateStory() {
  // Get the input word
  const inputWord = wordInput.value.trim();

  // Check if input word is empty
  if (inputWord === '') {
    alert('Please enter a word');
    return;
  }
  storyOutput.textContent += inputWord + ' ';
  wordInput.value = '';

  // Fetch the data from the API
  fetch(`http://127.0.0.1:8000/?word=${inputWord}`)
    .then(response => response.json())
    .then(data => {
      // Check if generated_word is an array
      if (Array.isArray(data.generated_word)) {
        // Join the generated words into a single string
        const story = data.generated_word.join(' ');

        // Display the story in the output box
        storyOutput.textContent += story + ' ';
      } else {
        // Handle the case when generated_word is not an array
        alert('Invalid response from the API');
      }
    })
    .catch(error => {
      console.log(error);
      alert('An error occurred');
    });
}
