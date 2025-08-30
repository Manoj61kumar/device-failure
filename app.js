// Medical Device Monitoring System - Connected with Backend ML

// Device Configuration
const DEVICE_MAPPING = {
    "Alaris GH": "Infusion Pump",
    "Baxter Flo-Gard": "Infusion Pump",
    "Smiths Medfusion": "Infusion Pump",
    "Baxter AK 96": "Dialysis Machine",
    "Fresenius 4008": "Dialysis Machine",
    "NxStage System One": "Dialysis Machine",
    "Datex Ohmeda S5": "Anesthesia Machine",
    "Drager Fabius Trio": "Anesthesia Machine",
    "GE Aisys": "Anesthesia Machine",
    "Drager V500": "Patient Ventilator",
    "Hamilton G5": "Patient Ventilator",
    "Puritan Bennett 980": "Patient Ventilator",
    "HeartStart FRx": "Defibrillator",
    "Lifepak 20": "Defibrillator",
    "Philips HeartStrart": "Defibrillator",
    "Zoll R Series": "Defibrillator",
    "GE Logiq E9": "Ultrasound Machine",
    "Philips EPIQ": "Ultrasound Machine",
    "Siemens Acuson": "Ultrasound Machine",
    "Siemens S2000": "Ultrasound Machine",
    "GE Revolution": "CT Scanner",
    "Philips Ingenuity": "CT Scanner",
    "GE MAC 2000": "ECG Monitor",
    "Phillips PageWriter": "ECG Monitor"
};

const DEVICE_TYPES = [
    "Anesthesia Machine",
    "CT Scanner",
    "Defibrillator",
    "Dialysis Machine",
    "ECG Monitor",
    "Infusion Pump",
    "Patient Ventilator",
    "Ultrasound Machine"
];
const LOCATIONS = [
    "Hospital A - ICU",
    "Hospital A - Emergency",
    "Hospital B - Nephrology",
    "Hospital B - Cardiology",
    "Hospital C - Surgery"
];

// Global App State
let devices = [];
let alerts = [];
let isStreaming = false;
let streamingInterval = null;
let charts = {};
let currentTheme = "light";
let updateInterval = 2000;
let soundEnabled = false;
let updateCount = 0;
let lastStats = { healthy: 0, warning: 0, critical: 0 };
let chartsPaused = { risk: false, temperature: false };

// --------------------- NEW: ML API CALL ----------------------
async function fetchPrediction(device) {
    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                DeviceType: device.deviceType,
                DeviceName: device.deviceName,
                RuntimeHours: device.runtimeHours,
                TemperatureC: device.temperature,
                PressureKPa: device.pressure,
                VibrationMM_S: device.vibration,
                CurrentDrawA: device.currentDraw,
                ErrorLogsCount: device.errorLogs,
                ApproxDeviceAgeYears: device.deviceAge,
                NumRepairs: device.repairs,
                Location: device.location
            })
        });
        const result = await response.json();
        device.prediction = result.prediction || "Low";
        device.confidence = 0.9; // placeholder
    } catch (error) {
        console.error("Prediction API error:", error);
        device.prediction = "Low"; // fallback
    }
}

// --------------------- Init ----------------------
document.addEventListener("DOMContentLoaded", function () {
    initializeApplication();
    setupEventListeners();
    generateDevices();
    setupCharts();
    startLiveStreaming();
});

function initializeApplication() {
    if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
        currentTheme = "dark";
        updateThemeIcon();
    }
    setupManualForm();
    setupInventoryFilters();
    updateStreamingControls();
}

// --------------------- Event Listeners ----------------------
function setupEventListeners() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", function (e) {
            e.preventDefault();
            const tabName = this.getAttribute("data-tab");
            if (tabName) switchTab(tabName);
        });
    });

    document.getElementById("toggleStreamBtn").addEventListener("click", toggleStreaming);
    document.getElementById("soundToggle").addEventListener("click", toggleSound);

    const speedSlider = document.getElementById("speedSlider");
    speedSlider.addEventListener("input", (e) => {
        updateInterval = parseInt(e.target.value) * 1000;
        document.getElementById("speedDisplay").textContent = e.target.value + "s";
        if (isStreaming) restartStreaming();
    });

    document.getElementById("pauseRiskChart").addEventListener("click", () => {
        chartsPaused.risk = !chartsPaused.risk;
        document.getElementById("pauseRiskChart").textContent = chartsPaused.risk ? "▶️" : "⏸️";
    });
    document.getElementById("pauseTempChart").addEventListener("click", () => {
        chartsPaused.temperature = !chartsPaused.temperature;
        document.getElementById("pauseTempChart").textContent = chartsPaused.temperature ? "▶️" : "⏸️";
    });

    document.getElementById("themeToggle").addEventListener("click", toggleTheme);
    document.getElementById("exportBtn").addEventListener("click", exportData);
    document.getElementById("runBatchBtn").addEventListener("click", runBatchPrediction);
    document.getElementById("manualForm").addEventListener("submit", handleManualPrediction);
    document.getElementById("searchInput").addEventListener("input", filterInventory);
    document.getElementById("typeFilter").addEventListener("change", filterInventory);
    document.getElementById("locationFilter").addEventListener("change", filterInventory);
    document.getElementById("closeModal").addEventListener("click", closeModal);
    document.getElementById("deviceModal").addEventListener("click", (e) => {
        if (e.target.id === "deviceModal") closeModal();
    });
    document.getElementById("clearAlertsBtn").addEventListener("click", clearAlerts);
}

// --------------------- Device Generation ----------------------
function generateDevices() {
    devices = [];
    const deviceNames = Object.keys(DEVICE_MAPPING);
    for (let i = 0; i < 24; i++) {
        const deviceName = deviceNames[i];
        const deviceType = DEVICE_MAPPING[deviceName];
        const location = LOCATIONS[Math.floor(Math.random() * LOCATIONS.length)];
        const device = {
            id: i + 1,
            deviceName,
            deviceType,
            location,
            alerted: false,
            lastUpdate: new Date(),
            temperature: 20 + Math.random() * 25,
            vibration: Math.random() * 1.2,
            errorLogs: Math.floor(Math.random() * 30),
            runtimeHours: 1000 + Math.random() * 8000,
            deviceAge: 0.5 + Math.random() * 5,
            repairs: Math.floor(Math.random() * 8),
            pressure: 80 + Math.random() * 100,
            currentDraw: 3 + Math.random() * 8
        };
        devices.push(device);
    }
}

// --------------------- Streaming ----------------------
function startLiveStreaming() {
    if (isStreaming) return;
    isStreaming = true;
    updateStreamingControls();
    updateCount = 0;
    streamingInterval = setInterval(() => {
        updateAllDevicesRealtime();
        updateLiveDashboard();
        updateCharts();
        checkForAlerts();
        updateStreamingMetrics();
        updateCount++;
    }, updateInterval);
}

function stopLiveStreaming() {
    if (!isStreaming) return;
    isStreaming = false;
    if (streamingInterval) {
        clearInterval(streamingInterval);
        streamingInterval = null;
    }
    updateStreamingControls();
}

function toggleStreaming() {
    if (isStreaming) stopLiveStreaming();
    else startLiveStreaming();
}

function restartStreaming() {
    if (isStreaming) {
        stopLiveStreaming();
        setTimeout(startLiveStreaming, 100);
    }
}

// --------------------- UPDATED: Device Updates ----------------------
async function updateAllDevicesRealtime() {
    for (const device of devices) {
        device.temperature += (Math.random() - 0.5) * 2;
        device.vibration += (Math.random() - 0.5) * 0.1;
        device.errorLogs += Math.floor(Math.random() * 3);
        device.runtimeHours += 0.5;
        device.pressure += (Math.random() - 0.5) * 5;
        device.currentDraw += (Math.random() - 0.5) * 0.5;

        device.temperature = Math.max(15, Math.min(45, device.temperature));
        device.vibration = Math.max(0, Math.min(1.2, device.vibration));
        device.errorLogs = Math.max(0, Math.min(50, device.errorLogs));
        device.pressure = Math.max(50, Math.min(250, device.pressure));
        device.currentDraw = Math.max(1, Math.min(15, device.currentDraw));
        device.lastUpdate = new Date();

        // 🔥 Call ML backend instead of local simulation
        await fetchPrediction(device);
    }
}

// --------------------- The rest (Charts, UI, Alerts, Inventory) ----------------------
// Keep your existing functions here unchanged:
// - updateLiveDashboard()
// - calculateDeviceStats()
// - updateDeviceGrid()
// - setupCharts(), updateCharts()
// - checkForAlerts(), showPopupAlert(), playAlertSound()
// - updateStreamingControls()
// - Inventory filtering, manual prediction, etc.
