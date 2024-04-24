document.addEventListener('DOMContentLoaded', function() {
    // Add animation when sections come into view
    const sections = document.querySelectorAll('.about-section, .workflow-section');

    function checkScroll() {
        sections.forEach(section => {
            const sectionTop = section.getBoundingClientRect().top;
            const windowHeight = window.innerHeight;
            
            if (sectionTop < windowHeight * 0.9) {
                section.classList.add('animate');
            } else {
                section.classList.remove('animate');
            }
        });
    }

    window.addEventListener('scroll', checkScroll);
    checkScroll(); // Check on initial load
});
