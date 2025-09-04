const CONFIG = {
    API_URL: "http://localhost:8000",
    WEBSOCKET_URL: "ws://localhost:8000/ws/video_feed",
    TRIGGER_AUDIO_URL: "/trigger_audio_analysis",
    GET_AUDIO_RESULT_URL: "/get_latest_audio_result",
    POLLING_INTERVAL_MS: 30000,
    DEFAULT_VIDEO_URLS: [
        "../demo/3_usa.mp4",
        "../demo/5_usa.mp4",
        "../demo/7_usa.mp4",
        "../demo/6_usa.mp4",
    ],
    DEFAULT_AUDIO_URL: "../demo/audio.m4a",
};

// --- Global State ---
const state = {
    show_detected: true,
    show_density: false,
    show_inactive: false,
    
    isStreaming: false,
    cameraWebSockets: [null, null, null, null],
    audioPollInterval: null,
    
    cameraUrls: [...CONFIG.DEFAULT_VIDEO_URLS],
    audioUrl: CONFIG.DEFAULT_AUDIO_URL,
    
    cameraStats: Array(4).fill().map(() => ({
        detected: 0,
        inactive: 0,
        dense_areas: 0
    })),

    lastVocalization: null,
    isAnalyzingAudio: false,
    lastAnalyzedAudioUrl: null,
    lastAnalysisTimestamp: null,

    lastInactiveAlert: {},
    INACTIVE_THRESHOLD: 0.15 //percentage
};

// --- DOM Elements ---
const $ = (selector) => document.querySelector(selector);

const DOMElements = { // Cache
    datetime: $('#datetime'),

    canvases: Array.from({ length: 4 }, (_, i) => $(`#video-canvas-${i}`)),
    contexts: [],

    toggleControls: $('#toggle-controls'),
    settingsButton: $('#settings-button'),

    detectedCount: $('#detected-count'),
    densityCount: $('#density-count'),
    inactiveCount: $('#inactive-count'),

    vocalizationContent: $('#vocalization-content'),

    systemLog: $('#system-log'),

    settingsModal: $('#settings-modal'),
    saveSettingsBtn: $('#save-settings-btn'),
    stopAllStreamsBtn: $('#stop-all-streams-btn'),
    camUrlInputs: Array.from({ length: 4 }, (_, i) => $(`#cam${i+1}-url`)),
    audioUrlInput: $('#audio-url'),
    
    exportModal: $('#export-modal'),
    openExportModalBtn: $('#open-export-modal-btn'),
    startDateInput: $('#start-date'),
    endDateInput: $('#end-date'),
    cameraSelect: $('#camera-select'),
    downloadCsvBtn: $('#download-csv-btn'),
};

// Init canvas contexts
DOMElements.contexts = DOMElements.canvases.map(canvas => 
    canvas ? canvas.getContext('2d') : null
);

// --- Utils Functions ---
function updateDateTime() {
    if (!DOMElements.datetime) return;
    
    DOMElements.datetime.textContent = new Date().toLocaleString('id-ID', {
        dateStyle: 'full',
        timeStyle: 'short'
    });
}

function addLog(message, type = 'info') {
    if (!DOMElements.systemLog) return;
    
    const timestamp = new Date().toLocaleTimeString('id-ID', { hour12: false });
    const typeClass = {
        info: 'text-slate-500',
        warning: 'text-amber-600',
        danger: 'text-red-600 font-semibold'
    }[type] || 'text-slate-500';
    
    const logEntry = document.createElement('p');
    logEntry.className = typeClass;
    logEntry.innerHTML = `<span class="font-semibold text-slate-400 mr-2">${timestamp}</span> ${message}`;
    
    DOMElements.systemLog.prepend(logEntry);
    
    if (DOMElements.systemLog.children.length > 30) {
        DOMElements.systemLog.removeChild(DOMElements.systemLog.lastChild);
    }
}

function updateToggleButtons() {
    const buttons = DOMElements.toggleControls.querySelectorAll('button');
    const activeColors = {
        show_detected: '#22c55e', // Green
        show_density: '#f97316',  // Orange
        show_inactive: '#ef4444'  // Red
    };
    
    buttons.forEach(button => {
        const control = button.dataset.control;
        const isActive = state[control];
        
        button.classList.toggle('active', isActive);
        button.setAttribute('aria-pressed', isActive);
        
        if (isActive) {
            const color = activeColors[control];
            button.style.backgroundColor = color;
            button.style.borderColor = color;
            button.style.color = '#ffffff';
        } else {
            button.style.backgroundColor = '';
            button.style.borderColor = '';
            button.style.color = '';
        }
    });
}

function updateDailyAnalysisUI() {
    const totals = state.cameraStats.reduce((acc, stats) => ({
        detected: acc.detected + (stats.detected || 0),
        inactive: acc.inactive + (stats.inactive || 0),
        denseAreas: acc.denseAreas + (stats.dense_areas || 0)
    }), { detected: 0, inactive: 0, denseAreas: 0 });
    
    if (DOMElements.detectedCount) DOMElements.detectedCount.textContent = totals.detected;
    if (DOMElements.inactiveCount) DOMElements.inactiveCount.textContent = totals.inactive;
    if (DOMElements.densityCount) DOMElements.densityCount.textContent = totals.denseAreas;
}

function displayAudioResults(data) {
    if (!DOMElements.vocalizationContent) return;
    
    if (!data) {
        DOMElements.vocalizationContent.innerHTML = `<p class="text-sm text-slate-500">No response from server.</p>`;
        return;
    }

    if (data.status === "analyzing") {
        DOMElements.vocalizationContent.innerHTML = `<p class="text-sm text-amber-600">Analisis vokalisasi sedang berlangsung...</p>`;
        return;
    }

    if (data.status === "no_data" || data.prediction === null) {
        DOMElements.vocalizationContent.innerHTML = `<p class="text-sm text-slate-500">Menunggu analisis vokalisasi berlangsung...</p>`;
        return;
    }
    
    if (data.prediction === "Error") {
        DOMElements.vocalizationContent.innerHTML = `<p class="text-sm text-red-600">Error during audio analysis.</p>`;
        return;
    }
    
    const probabilities = data.probabilities || {};
    const statusMap = {
        'Healthy': { text: 'Sehat', color: 'green' },
        'Unhealthy': { text: 'Tidak Sehat', color: 'red' },
        'Noise': { text: 'Bising', color: 'amber' },
    };
    
    const dominantStatus = statusMap[data.prediction] || { 
        text: data.prediction, 
        color: 'slate' 
    };

    if(
        (data.prediction === "Unhealthy" || dominantStatus.text === "Tidak Sehat") &&
        state.lastVocalization !== "Unhealthy"
    ){
        addLog("Status Vokal: <strong>Tidak Sehat Terdeteksi</strong>", "danger");
        state.lastVocalization = "Unhealthy";
    }
    else if (data.prediction === "Healthy"){
        state.lastVocalization = "Healthy";
    }
    
    let barsHtml = '';
    for (const [key, value] of Object.entries(probabilities)) {
        const status = statusMap[key] || { text: key, color: 'slate' };
        const percentage = (value * 100).toFixed(1);
        
        barsHtml += `
            <div>
                <div class="flex justify-between text-xs mb-1">
                    <span class="font-medium text-${status.color}-700">${status.text}</span>
                    <span>${percentage}%</span>
                </div>
                <div class="w-full bg-slate-200 rounded-full h-2">
                    <div class="bg-${status.color}-500 h-2 rounded-full" style="width:${percentage}%"></div>
                </div>
            </div>
        `;
    }

    DOMElements.vocalizationContent.innerHTML = `
        <div class="flex items-center justify-between mb-3">
            <span class="text-slate-500 text-xs">Status Dominan:</span>
            <span class="font-bold text-base text-${dominantStatus.color}-600">${dominantStatus.text}</span>
        </div>
        <div class="space-y-2">${barsHtml}</div>
    `;


}

// --- Modal function ---
function openModal(modalEl) {
    modalEl.classList.add('show');
    modalEl.setAttribute('aria-hidden', 'false');
    document.body.classList.add('overflow-hidden');
}

function closeModal(modalEl) {
    modalEl.classList.remove('show');
    modalEl.setAttribute('aria-hidden', 'true');
    document.body.classList.remove('overflow-hidden');
}

function populateSettingsModal() {
    DOMElements.camUrlInputs.forEach((input, i) => {
        if (input) input.value = state.cameraUrls[i] || '';
    });
    
    if (DOMElements.audioUrlInput) {
        DOMElements.audioUrlInput.value = state.audioUrl || '';
    }
}

// Make Local Storage for url when change
function loadSettingsFromStorage() {
    const savedCameraUrls = localStorage.getItem('chickSenseCameraUrls');
    const savedAudioUrl = localStorage.getItem('chickSenseAudioUrl');

    if (savedCameraUrls) {
        try {
            state.cameraUrls = JSON.parse(savedCameraUrls);
        } catch (e) {
            console.error("Failed to parse saved camera URLs, using defaults.", e);
            state.cameraUrls = [...CONFIG.DEFAULT_VIDEO_URLS];
        }
    }
    if (savedAudioUrl) {
        state.audioUrl = savedAudioUrl;
    }
    addLog('Settings dimuat dari penyimpanan browser', 'info');
}

// --- Websocket Streaming functions ---
function connectWebSocket(cameraIndex) {
    const videoUrl = state.cameraUrls[cameraIndex];
    if (!videoUrl) {
        console.warn(`No URL for Camera ${cameraIndex + 1}, skipping connection.`);
        return;
    }
    
    if (state.cameraWebSockets[cameraIndex]) {
        state.cameraWebSockets[cameraIndex].close();
    }
    
    try {
        const ws = new WebSocket(CONFIG.WEBSOCKET_URL);
        ws.binaryType = "blob";
        state.cameraWebSockets[cameraIndex] = ws;
        
        ws.onopen = () => {
            addLog(`Menghubungkan ke Kamera ${cameraIndex + 1}`, 'info');
            
            ws.send(JSON.stringify({
                type: 'start_stream',
                video_url: videoUrl,
                audio_url: state.audioUrl,
                show_detected: state.show_detected,
                show_density: state.show_density,
                show_inactive: state.show_inactive,
                camera_id: cameraIndex + 1
            }));
            
            state.isStreaming = true;
        };
        
        ws.onmessage = (event) => {
            if (event.data instanceof Blob) {
                handleBlobMessage(event.data, cameraIndex);
            } else if (typeof event.data === 'string') {
                handleTextMessage(event.data, cameraIndex);
            }
        };
        
        ws.onclose = () => {
            addLog(`Kamera ${cameraIndex + 1} terputus.`, 'warning');
            
            const ctx = DOMElements.contexts[cameraIndex];
            if (ctx) ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            
            state.cameraWebSockets[cameraIndex] = null;
            state.cameraStats[cameraIndex] = { detected: 0, inactive: 0, dense_areas: 0 };
            updateDailyAnalysisUI();
            
            state.isStreaming = state.cameraWebSockets.some(ws => ws !== null);
            if (!state.isStreaming) {
                addLog("Semua streams terputus.", 'danger');
            }
        };
        
        ws.onerror = (error) => {
            console.error(`WebSocket Error for Camera ${cameraIndex + 1}:`, error);
            addLog(`Koneksi gagal untuk Kamera ${cameraIndex + 1}.`, 'danger');
        };
    } catch (error) {
        console.error(`Failed to create WebSocket for Camera ${cameraIndex + 1}:`, error);
        addLog(`Gagal terhubung ke Kamera ${cameraIndex + 1}.`, 'danger');
    }
}

function handleBlobMessage(blob, cameraIndex) {
    const imageUrl = URL.createObjectURL(blob);
    const img = new Image();
    
    img.onload = () => {
        const ctx = DOMElements.contexts[cameraIndex];
        if (ctx) {
            ctx.canvas.width = img.width;
            ctx.canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        }
        URL.revokeObjectURL(imageUrl);
    };
    
    img.onerror = () => {
        console.error(`Failed to load image for Camera ${cameraIndex + 1}`);
        URL.revokeObjectURL(imageUrl);
    };
    
    img.src = imageUrl;
}

function handleTextMessage(data, cameraIndex) {
    try {
        const parsedData = JSON.parse(data);
        
        if (parsedData.type === 'stats') {
            const stats = {
                detected: Number(parsedData.detected || 0),
                inactive: Number(parsedData.inactive || 0),
                dense_areas: Number(parsedData.dense_areas || 0),
            };

            state.cameraStats[cameraIndex] = stats;
            updateDailyAnalysisUI();

            if (stats.detected > 0) {
                const inactiveRatio = stats.inactive / stats.detected;
                const percent = Math.round(inactiveRatio * 100);
                const cameraId = cameraIndex + 1;

                const isAboveThreshold = inactiveRatio > state.INACTIVE_THRESHOLD;
                const hasAlerted = state.lastInactiveAlert[cameraId];

                if (isAboveThreshold && !hasAlerted){
                    const message = `Camera ${cameraId}: <strong> Persentase ayam tidak aktif cukup tinggi (${percent}%) </strong>`;

                    addLog(message, "danger");
                    state.lastInactiveAlert[cameraId] = true;
                }
                else if (!isAboveThreshold && hasAlerted){
                    state.lastInactiveAlert[cameraId] = false;
                }
            }

        }else if (parsedData.type === 'status') {
            const msg = parsedData.message;
            if (msg === "Display settings updated") {
                return;
            }
            addLog(`[Global] ${msg}`, 'info');
        }
    } catch (e) {
        console.error("Error parsing WebSocket JSON message:", e, data);
    }
}

function startAllStreams() {
    addLog("Mencoba memulai semua stream yang terkonfigurasi...", 'info');
    
    for (let i = 0; i < 4; i++) {
        setTimeout(() => connectWebSocket(i), i * 100);
    }
    
    if (!state.audioPollInterval) {
        fetchLatestAudioResult();
        state.audioPollInterval = setInterval(
            fetchLatestAudioResult, 
            CONFIG.POLLING_INTERVAL_MS
        );
    }
}

function stopAllStreams() {
    addLog("Menghentikan semua stream...", 'warning');
    
    state.cameraWebSockets.forEach((ws, index) => {
        if (ws) {
            ws.close();
            state.cameraWebSockets[index] = null;
        }
    });
    
    if (state.audioPollInterval) {
        clearInterval(state.audioPollInterval);
        state.audioPollInterval = null;
    }
    
    displayAudioResults(null);
    state.isStreaming = false;
}

async function fetchLatestAudioResult() {
    try {
        const response = await fetch(`${CONFIG.API_URL}${CONFIG.GET_AUDIO_RESULT_URL}`);
        
        if (!response.ok) {
            if(response.status === 404){
                return;
            }throw new Error(`Server status ${response.status}`)
        }
        
        const audioData = await response.json();

        const resultKey = JSON.stringify(audioData);
        const wasAnalyzing = state.isAnalyzingAudio;

        if (state.lastAudioResultKey && state.lastAudioResultKey === resultKey){
            return;
        }

        state.lastAudioResultKey = resultKey;

        if (wasAnalyzing){
            addLog("Hasil vokalisasi diperbarui", "info");
            state.isAnalyzingAudio = false;
            state.lastAnalysisTimestamp = new Date();
        }else{
            addLog("Hasil vokalisasi tersedia", "info");
        }


        displayAudioResults(audioData);
    } catch (error) {
        console.error("Error fetching audio result:", error);
        
        if (state.isAnalyzingAudio){
            addLog("Analisis audio gagal", "danger");
            state.isAnalyzingAudio = false;
        }
    }
}

async function downloadCSV() {
    const start = DOMElements.startDateInput.value;
    const end = DOMElements.endDateInput.value;
    const cameraId = DOMElements.cameraSelect.value;
    
    if (!start || !end) {
        addLog("Silakan pilih tanggal mulai dan tanggal selesai untuk ekspor.", 'warning');
        return;
    }
    
    try {
        const url = new URL(`${CONFIG.API_URL}/metrics/export`);
        url.searchParams.set('start', start);
        url.searchParams.set('end', end);
        if (cameraId) url.searchParams.set('camera_id', cameraId);
        
        addLog(`Exporting CSV for ${start} to ${end}...`, 'info');
        
        const response = await fetch(url.toString());
        if (!response.ok) {
            throw new Error(`Export failed with status ${response.status}`);
        }
        
        const blob = await response.blob();
        const downloadUrl = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = `metrics_${start}_to_${end}${cameraId ? `_cam${cameraId}` : ''}.csv`;
        document.body.appendChild(a);
        a.click();
        
        // Clean up
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(downloadUrl);
        }, 100);
        
        addLog("CSV download started.", 'info');
        closeModal(DOMElements.exportModal);
    } catch (error) {
        console.error("CSV Download Error:", error);
        addLog(error.message, 'danger');
    }
}

// -- Event Handlers --
function handleToggleControlClick(e) {
    const button = e.target.closest('button');
    if (!button) return;
    
    const control = button.dataset.control;
    if (!control) return;
    
    state[control] = !state[control];
    updateToggleButtons();

    const labelMap = {
        show_detected: 'Deteksi',
        show_density: 'Kepadatan',
        show_inactive: 'Inaktivitas'
    };

    const displayName = labelMap[control] || control;
    addLog(`Tampilan diperbarui: ${displayName} ${state[control] ? 'dinyalakan' : 'dimatikan'}`, 'info');

    const payload = JSON.stringify({
        type: 'display_settings_update',
        show_detected: state.show_detected,
        show_density: state.show_density,
        show_inactive: state.show_inactive
    });
    
    state.cameraWebSockets.forEach(ws => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(payload);
        }
    });
}

function handleSettingsButtonClick() {
    addLog('Tip: Double-click pada ikon gear untuk membuka setting kamera.', 'info');
}

function handleSettingsButtonDoubleClick() {
    populateSettingsModal();
    openModal(DOMElements.settingsModal);
}

function handleSaveSettingsClick() {
    state.cameraUrls = DOMElements.camUrlInputs.map(input => 
        input ? input.value.trim() : ''
    );
    state.audioUrl = DOMElements.audioUrlInput ? 
        DOMElements.audioUrlInput.value.trim() : '';
    
    try {
        localStorage.setItem('chickSenseCameraUrls', JSON.stringify(state.cameraUrls));
        localStorage.setItem('chickSenseAudioUrl', state.audioUrl);
        addLog('Pengaturan disimpan ke penyimpanan peramban.', 'info');
    } catch (e) {
        console.error("Failed to save settings to local storage:", e);
        addLog('Tidak dapat menyimpan pengaturan.', 'danger');
    }

    addLog('Pengaturan disimpan. Memulai ulang aliran...', 'info');
    closeModal(DOMElements.settingsModal);
    
    stopAllStreams();
    setTimeout(startAllStreams, 500);
}

function handleStopStreamsClick() {
    stopAllStreams();
    closeModal(DOMElements.settingsModal);
}

function handleExportButtonClick() {
    addLog('Tip: Double-click pada "Export CSV" untuk mengekspor metric ke CSV.', 'info');
}

function handleExportButtonDoubleClick() {
    openModal(DOMElements.exportModal);
}

function handleModalCloseClick(e) {
    const modal = e.currentTarget;
    
    if (e.target.classList.contains('modal-backdrop') || 
        e.target.closest('.modal-close-btn')) {
        closeModal(modal);
    }
}

function handleEscapeKey(e) {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal.show').forEach(closeModal);
    }
}

// -- Add Event Listener --
function setupEventListeners() {
    if (DOMElements.toggleControls) {
        DOMElements.toggleControls.addEventListener('click', handleToggleControlClick);
    }
    
    if (DOMElements.settingsButton) {
        DOMElements.settingsButton.addEventListener('click', handleSettingsButtonClick);
        DOMElements.settingsButton.addEventListener('dblclick', handleSettingsButtonDoubleClick);
    }
    
    if (DOMElements.saveSettingsBtn) {
        DOMElements.saveSettingsBtn.addEventListener('click', handleSaveSettingsClick);
    }
    
    if (DOMElements.stopAllStreamsBtn) {
        DOMElements.stopAllStreamsBtn.addEventListener('click', handleStopStreamsClick);
    }
    
    if (DOMElements.openExportModalBtn) {
        DOMElements.openExportModalBtn.addEventListener('click', handleExportButtonClick);
        DOMElements.openExportModalBtn.addEventListener('dblclick', handleExportButtonDoubleClick);
    }
    
    if (DOMElements.downloadCsvBtn) {
        DOMElements.downloadCsvBtn.addEventListener('click', downloadCSV);
    }
    
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', handleModalCloseClick);
    });
    
    window.addEventListener('keydown', handleEscapeKey);
}

// -- Init --
function initialize() {
    updateDateTime();
    updateToggleButtons();
    updateDailyAnalysisUI();
    displayAudioResults(null);
    loadSettingsFromStorage();

    setupEventListeners();

    const today = new Date().toISOString().split('T')[0];
    if (DOMElements.startDateInput) DOMElements.startDateInput.value = today;
    if (DOMElements.endDateInput) DOMElements.endDateInput.value = today;
    
    setInterval(updateDateTime, 30000); 
    
    addLog('Sistem diinisialisasi. Selamat datang! IKGC', 'info');
    startAllStreams();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
} else {
    initialize();
}


