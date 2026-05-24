/**
 * Распознавание рукописных символов
 * Модель: improved_cnn_v5 (EMNIST Balanced, 47 классов)
 * Инференс: ONNX Runtime Web (in-browser)
 */

// ============ КОНФИГУРАЦИЯ ============

const CONFIG = {
    modelPath: 'model_v5.onnx',
    canvasSize: 280,
    modelInputSize: 28,
    mean: 0.1307,
    std: 0.3081,
    topK: 3,
};

// Маппинг 47 классов EMNIST Balanced
const EMNIST_BALANCED_CLASSES = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
];

// ============ ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ============

let ortSession = null;
let isDrawing = false;
let isModelLoaded = false;

const drawCanvas = document.getElementById('drawCanvas');
const drawCtx = drawCanvas.getContext('2d');
const previewCanvas = document.getElementById('previewCanvas');
const previewCtx = previewCanvas.getContext('2d');
const recognizeBtn = document.getElementById('recognizeBtn');
const clearBtn = document.getElementById('clearBtn');
const transposeToggle = document.getElementById('transposeToggle');
const loadingIndicator = document.getElementById('loadingIndicator');
const errorMessage = document.getElementById('errorMessage');
const resultsDiv = document.getElementById('results');
const predictionsDiv = document.getElementById('predictions');
const inferenceTimeDiv = document.getElementById('inferenceTime');

// ============ ИНИЦИАЛИЗАЦИЯ ============

function initCanvas() {
    drawCtx.fillStyle = 'black';
    drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    drawCtx.strokeStyle = 'white';
    drawCtx.lineWidth = 14;
    drawCtx.lineCap = 'round';
    drawCtx.lineJoin = 'round';
    
    previewCtx.fillStyle = 'black';
    previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    previewCtx.imageSmoothingEnabled = false;
    
    // Включаем транспонирование по умолчанию для EMNIST
    transposeToggle.checked = true;
    console.log('🔄 Транспонирование EMNIST включено по умолчанию (можно отключить)');
}

// ============ ОБРАБОТЧИКИ РИСОВАНИЯ ============

function getCanvasCoords(e) {
    const rect = drawCanvas.getBoundingClientRect();
    const scaleX = drawCanvas.width / rect.width;
    const scaleY = drawCanvas.height / rect.height;
    
    let clientX, clientY;
    
    if (e.touches && e.touches.length > 0) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
    } else if (e.changedTouches && e.changedTouches.length > 0) {
        clientX = e.changedTouches[0].clientX;
        clientY = e.changedTouches[0].clientY;
    } else {
        clientX = e.clientX;
        clientY = e.clientY;
    }
    
    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY
    };
}

function startDrawing(e) {
    isDrawing = true;
    const coords = getCanvasCoords(e);
    drawCtx.beginPath();
    drawCtx.moveTo(coords.x, coords.y);
    e.preventDefault();
}

function draw(e) {
    if (!isDrawing) return;
    const coords = getCanvasCoords(e);
    drawCtx.lineTo(coords.x, coords.y);
    drawCtx.stroke();
    e.preventDefault();
}

function stopDrawing(e) {
    if (isDrawing) {
        isDrawing = false;
        drawCtx.closePath();
    }
    if (e) e.preventDefault();
}

// Мышь
drawCanvas.addEventListener('mousedown', startDrawing);
drawCanvas.addEventListener('mousemove', draw);
drawCanvas.addEventListener('mouseup', stopDrawing);
drawCanvas.addEventListener('mouseleave', stopDrawing);

// Сенсорный экран
drawCanvas.addEventListener('touchstart', startDrawing, { passive: false });
drawCanvas.addEventListener('touchmove', draw, { passive: false });
drawCanvas.addEventListener('touchend', stopDrawing);

// ============ ЗАГРУЗКА МОДЕЛИ ============

async function loadModel() {
    if (isModelLoaded) return;
    
    try {
        loadingIndicator.style.display = 'flex';
        errorMessage.style.display = 'none';
        recognizeBtn.disabled = true;
        
        console.log('🔄 Загрузка модели ONNX...');
        
        ortSession = await ort.InferenceSession.create(CONFIG.modelPath, {
            executionProviders: ['wasm']
        });
        
        isModelLoaded = true;
        recognizeBtn.disabled = false;
        loadingIndicator.style.display = 'none';
        
        console.log('✅ Модель загружена успешно');
        console.log('Входы:', ortSession.inputNames);
        console.log('Выходы:', ortSession.outputNames);
        
        // Показываем уведомление о готовности
        showModelReady();
        
    } catch (error) {
        console.error('❌ Ошибка загрузки:', error);
        loadingIndicator.style.display = 'none';
        errorMessage.style.display = 'block';
        errorMessage.innerHTML = `
            <strong>Ошибка загрузки модели</strong><br>
            ${error.message}<br>
            <small>Проверьте консоль (F12) для деталей</small>
        `;
    }
}

function showModelReady() {
    const status = document.getElementById('status');
    const readyMsg = document.createElement('div');
    readyMsg.textContent = '✅ Модель готова к распознаванию';
    readyMsg.style.cssText = `
        color: #10b981;
        font-weight: 600;
        text-align: center;
        padding: 8px;
        background: #f0fdf4;
        border-radius: 8px;
        margin-bottom: 12px;
        animation: fadeIn 0.5s;
    `;
    status.appendChild(readyMsg);
    
    setTimeout(() => {
        readyMsg.remove();
    }, 3000);
}

// ============ ПРЕПРОЦЕССИНГ ============

function preprocessCanvas() {
    const imageData = drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = CONFIG.modelInputSize;
    tempCanvas.height = CONFIG.modelInputSize;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.imageSmoothingEnabled = true;
    tempCtx.imageSmoothingQuality = 'high';
    
    tempCtx.drawImage(drawCanvas, 0, 0, CONFIG.modelInputSize, CONFIG.modelInputSize);
    const tempImageData = tempCtx.getImageData(0, 0, CONFIG.modelInputSize, CONFIG.modelInputSize);
    
    const pixels = new Float32Array(CONFIG.modelInputSize * CONFIG.modelInputSize);
    
    for (let i = 0; i < CONFIG.modelInputSize * CONFIG.modelInputSize; i++) {
        const pixelValue = tempImageData.data[i * 4];
        pixels[i] = (pixelValue / 255.0 - CONFIG.mean) / CONFIG.std;
    }
    
    if (transposeToggle.checked) {
        const transposed = new Float32Array(CONFIG.modelInputSize * CONFIG.modelInputSize);
        for (let i = 0; i < CONFIG.modelInputSize; i++) {
            for (let j = 0; j < CONFIG.modelInputSize; j++) {
                transposed[j * CONFIG.modelInputSize + i] = pixels[i * CONFIG.modelInputSize + j];
            }
        }
        return { tensor: transposed, imageData: tempImageData };
    }
    
    return { tensor: pixels, imageData: tempImageData };
}

// ============ ОТОБРАЖЕНИЕ ПРЕДПРОСМОТРА ============

function updatePreview(imageData) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = CONFIG.modelInputSize;
    tempCanvas.height = CONFIG.modelInputSize;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.putImageData(imageData, 0, 0);
    
    previewCtx.imageSmoothingEnabled = false;
    previewCtx.fillStyle = 'black';
    previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    previewCtx.drawImage(tempCanvas, 0, 0, previewCanvas.width, previewCanvas.height);
}

// ============ ИНФЕРЕНС ============

async function runInference() {
    if (!isModelLoaded) {
        errorMessage.style.display = 'block';
        errorMessage.textContent = 'Модель ещё не загружена. Подождите...';
        return;
    }
    
    try {
        const startTime = performance.now();
        
        const { tensor, imageData } = preprocessCanvas();
        updatePreview(imageData);
        
        console.log('📊 Статистика тензора:',
            'мин:', Math.min(...tensor).toFixed(3),
            'макс:', Math.max(...tensor).toFixed(3)
        );
        
        const inputTensor = new ort.Tensor('float32', tensor, [1, 1, CONFIG.modelInputSize, CONFIG.modelInputSize]);
        
        const outputs = await ortSession.run({ input: inputTensor });
        const logits = Array.from(outputs.output.data);
        
        const probabilities = softmax(logits);
        const topK = getTopK(probabilities, CONFIG.topK);
        
        const endTime = performance.now();
        
        console.log('🏆 Топ-3:', topK.map(({index, prob}) => 
            `${EMNIST_BALANCED_CLASSES[index]}=${(prob*100).toFixed(1)}%`
        ).join(', '));
        
        displayResults(topK, (endTime - startTime).toFixed(1));
        
    } catch (error) {
        console.error('❌ Ошибка инференса:', error);
        errorMessage.style.display = 'block';
        errorMessage.textContent = `Ошибка: ${error.message}`;
        resultsDiv.style.display = 'none';
    }
}

// ============ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ============

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const exps = logits.map(x => Math.exp(x - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sumExps);
}

function getTopK(probabilities, k) {
    return probabilities
        .map((prob, index) => ({ index, prob }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, k);
}

function displayResults(topK, inferenceTime) {
    errorMessage.style.display = 'none';
    resultsDiv.style.display = 'block';
    
    predictionsDiv.innerHTML = topK.map(({ index, prob }) => {
        const symbol = EMNIST_BALANCED_CLASSES[index];
        const percent = (prob * 100).toFixed(1);
        const width = Math.max(prob * 100, 0.5);
        
        // Цвет полоски зависит от уверенности
        const barColor = prob > 0.7 
            ? 'linear-gradient(135deg, #10b981, #059669)'  // Зеленый - высокая уверенность
            : prob > 0.3 
                ? 'linear-gradient(135deg, #2563eb, #7c3aed)'  // Синий - средняя
                : 'linear-gradient(135deg, #64748b, #475569)';  // Серый - низкая
        
        return `
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;padding:8px;background:#f8fafc;border-radius:8px;border:1px solid #e2e8f0;">
                <div style="font-size:28px;font-weight:700;width:45px;text-align:center;color:#1e293b;">${symbol}</div>
                <div style="flex:1;background:#e5e7eb;border-radius:6px;height:28px;overflow:hidden;">
                    <div style="width:${width}%;height:100%;background:${barColor};border-radius:6px;transition:width 0.5s;min-width:${width > 0 ? '3px' : '0'};"></div>
                </div>
                <div style="font-weight:600;font-size:15px;color:#1e293b;min-width:55px;text-align:right;">${percent}%</div>
            </div>
        `;
    }).join('');
    
    inferenceTimeDiv.textContent = `⚡ Время распознавания: ${inferenceTime} мс`;
}

// ============ ОЧИСТКА ============

function clearCanvas() {
    drawCtx.fillStyle = 'black';
    drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    previewCtx.fillStyle = 'black';
    previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    resultsDiv.style.display = 'none';
    errorMessage.style.display = 'none';
}

// ============ ОБРАБОТЧИКИ ============

recognizeBtn.addEventListener('click', runInference);
clearBtn.addEventListener('click', clearCanvas);

// Горячие клавиши
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        runInference();
    } else if (e.key === 'Delete' || (e.ctrlKey && e.key === 'Backspace')) {
        e.preventDefault();
        clearCanvas();
    }
});

// ============ ЗАПУСК ============

window.addEventListener('load', () => {
    console.log('🚀 Инициализация...');
    console.log('ℹ️ Подсказка: Enter - распознать, Delete - очистить');
    initCanvas();
    loadModel();
});