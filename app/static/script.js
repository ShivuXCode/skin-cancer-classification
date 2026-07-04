document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const analyzeBtn = document.getElementById("analyze-btn");
    const resetBtn = document.getElementById("reset-btn");
    const loading = document.getElementById("loading");
    const results = document.getElementById("results");

    let selectedFile = null;

    // --- Drag and Drop Logic ---
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drop-zone--over");
    });

    ["dragleave", "dragend"].forEach(type => {
        dropZone.addEventListener(type, (e) => {
            dropZone.classList.remove("drop-zone--over");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drop-zone--over");

        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please upload an image file.");
            return;
        }

        selectedFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add("hidden");
            previewContainer.classList.remove("hidden");
            results.classList.add("hidden");
        };
        reader.readAsDataURL(file);
    }

    // --- Reset ---
    resetBtn.addEventListener("click", () => {
        selectedFile = null;
        fileInput.value = "";
        previewContainer.classList.add("hidden");
        results.classList.add("hidden");
        dropZone.classList.remove("hidden");
        
        // Reset bars
        document.getElementById("bar-primary").style.width = "0%";
        document.getElementById("bar-specific").style.width = "0%";
    });

    // --- API Call ---
    analyzeBtn.addEventListener("click", async () => {
        if (!selectedFile) return;

        // UI State
        analyzeBtn.classList.add("hidden");
        resetBtn.classList.add("hidden");
        loading.classList.remove("hidden");
        results.classList.add("hidden");

        const formData = new FormData();
        formData.append("image", selectedFile);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            loading.classList.add("hidden");
            analyzeBtn.classList.remove("hidden");
            resetBtn.classList.remove("hidden");

            if (data.error) {
                alert("Error: " + data.error);
                return;
            }

            // Populate Results
            document.getElementById("res-primary").textContent = data.primary_diagnosis;
            document.getElementById("conf-primary").textContent = data.primary_confidence;
            
            // Set Color and Bar based on Diagnosis
            const resPrimary = document.getElementById("res-primary");
            const barPrimary = document.getElementById("bar-primary");
            
            if (data.primary_diagnosis.toLowerCase() === "malignant") {
                resPrimary.style.color = "var(--danger-color)";
                barPrimary.style.backgroundColor = "var(--danger-color)";
            } else {
                resPrimary.style.color = "var(--success-color)";
                barPrimary.style.backgroundColor = "var(--success-color)";
            }

            results.classList.remove("hidden");

            // Animate Bars after a slight delay
            setTimeout(() => {
                document.getElementById("bar-primary").style.width = data.primary_confidence;
            }, 100);

        } catch (err) {
            loading.classList.add("hidden");
            analyzeBtn.classList.remove("hidden");
            resetBtn.classList.remove("hidden");
            alert("Failed to connect to the server.");
            console.error(err);
        }
    });
});
