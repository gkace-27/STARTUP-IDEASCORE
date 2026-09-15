const form = document.querySelector("#score-form");
const result = document.querySelector("#result");

form.addEventListener("submit", async (event) => {
	event.preventDefault();
	const button = form.querySelector("button");
	button.disabled = true;
	button.firstChild.textContent = "Scoring startup... ";

	try {
		const response = await fetch("/predict", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(Object.fromEntries(new FormData(form))),
		});
		const data = await response.json();
		if (!response.ok) throw new Error(data.error || "Unable to score startup");
		result.innerHTML = `<strong>${data.label}</strong>${data.probability}% estimated success likelihood`;
		result.classList.add("visible");
	} catch (error) {
		result.textContent = error.message;
		result.classList.add("visible");
	} finally {
		button.disabled = false;
		button.firstChild.textContent = "Calculate startup score ";
	}
});
