// Cấu hình API cho frontend
const API_CONFIG = {
    // IP của máy chủ API (máy 1 - có GPU)
    API_BASE_URL: "http://192.168.0.112:8000",
    
    // Các endpoint
    CHAT_ENDPOINT: "/chat",
    CHAT_STREAM_ENDPOINT: "/chat/stream",
    HEALTH_ENDPOINT: "/health",
    EMERGENCY_ENDPOINT: "/emergency",
    SEMANTIC_SEARCH_ENDPOINT: "/semantic_search"
};

// Hàm lấy URL đầy đủ cho endpoint
function getApiUrl(endpoint) {
    return API_CONFIG.API_BASE_URL + endpoint;
}

// Export cho sử dụng trong HTML
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { API_CONFIG, getApiUrl };
} 