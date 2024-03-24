// Script Loading
const AUTHOR = "Adarsh Maurya";
const VERSION = "1.0.0";
const WEBSITE = "https://adarshmaurya.onionreads.com/";

console.log(`%cScript Loaded Successfully\n%cAuthor: ${AUTHOR}\nVersion: ${VERSION}\nWebsite: ${WEBSITE}`);

let navDemo = document.getElementById("nav-demo");
let navDemoSpan = document.getElementById("nav-demo-span");

navDemo.addEventListener("mouseover", () => {
  navDemoSpan.style.transform = "translateX(2rem)";
  navDemo.style.transition ="all 0.3s ease";

});