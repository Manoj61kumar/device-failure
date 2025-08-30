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
        device.confidence = 0.9; // optional placeholder
    } catch (error) {
        console.error("Prediction API error:", error);
        device.prediction = "Low"; // fallback if API fails
    }
}
