// Script Loading
const AUTHOR = "Adarsh Maurya";
const VERSION = "1.0.0";
const WEBSITE = "https://adarshmaurya.onionreads.com/";

console.log(`%cScript Loaded Successfully\n%cAuthor: ${AUTHOR}\nVersion: ${VERSION}\nWebsite: ${WEBSITE}`);

let navDemo = document.getElementById("nav-demo");
let navDemoSpan = document.getElementById("nav-demo-span");

navDemo.addEventListener("mouseover", () => {
  navDemoSpan.style.transform = "translateX(2rem)";
  navDemoSpan.style.transition ="all 0.3s ease";

});

//Typiing animation
// make a function that will take an element and then add typing animation to it's text content

let typeElement = document.getElementById("dynamic-text");
console.log(typeElement.innerText);

function typeWriterAnimation(elementId, options = {}) {
  const element = document.getElementById(elementId);
  if (!element) {
      console.error(`Element with ID "${elementId}" not found.`);
      return;
  }

  const defaultOptions = {
      speed: 50, // Typing speed in milliseconds
      delay: 1000, // Delay before typing starts in milliseconds
      cursor: '|', // Cursor character
      cursorSpeed: 400, // Cursor blink speed in milliseconds
      loop: false // Whether to loop the animation
  };

  const config = { ...defaultOptions, ...options };

  const text = element.innerText.trim();
  element.innerText = '';

  let index = 0;
  let direction = 1;
  let isDeleting = false;

  function animateTyping() {
      setTimeout(() => {
          if (index <= text.length && !isDeleting) {
              element.innerText = text.substring(0, index) + config.cursor;
              index++;
          } else if (index >= 0 && isDeleting) {
              element.innerText = text.substring(0, index) + config.cursor;
              index--;
          }

          if (index === text.length + 1) {
              isDeleting = true;
              direction = -1;
          } else if (index === 0 && isDeleting) {
              isDeleting = false;
              direction = 1;

              if (!config.loop) return;
          }

          animateTyping();
      }, config.speed);
  }

  setTimeout(animateTyping, config.delay);

  if (config.cursor) {
      const cursorInterval = setInterval(() => {
          if (element.innerText.endsWith(config.cursor)) {
              element.innerText = element.innerText.slice(0, -1);
          } else {
              element.innerText += config.cursor;
          }
      }, config.cursorSpeed);

      // Stop cursor blinking when animation ends
      setTimeout(() => clearInterval(cursorInterval), config.delay + (text.length + 1) * config.speed);
  }
}

// Example usage:
typeWriterAnimation("dynamic-text", {
  speed: 100,
  delay: 100,
  cursor: '|',
  cursorSpeed: 200,
  loop: true
});