"""
Floating AI Assistant Component for the dashboard.
"""

import streamlit as st


def inject_floating_orb():
    """Inject floating orb component into the Streamlit app"""

    # Use st.markdown with unsafe_allow_html to inject CSS and HTML
    floating_orb_style = """
    <style>
    /* Floating orb container */
    .floating-orb-container {
        position: fixed !important;
        bottom: 30px !important;
        right: 30px !important;
        z-index: 999999 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif !important;
    }

    /* The floating orb */
    .floating-orb {
        width: 60px !important;
        height: 60px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 50% !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: none !important;
        outline: none !important;
        animation: float 3s ease-in-out infinite !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .floating-orb:before {
        content: '' !important;
        position: absolute !important;
        top: 10% !important;
        left: 10% !important;
        width: 80% !important;
        height: 80% !important;
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 50% !important;
    }

    .floating-orb:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4) !important;
    }

    .floating-orb:active {
        transform: scale(0.95) !important;
    }

    /* Orb icon */
    .orb-icon {
        color: white !important;
        font-size: 24px !important;
        z-index: 1 !important;
        position: relative !important;
    }

    /* Floating animation */
    @keyframes float {
        0%, 100% {
            transform: translateY(0px) !important;
        }
        50% {
            transform: translateY(-10px) !important;
        }
    }

    /* Popup window */
    .floating-popup {
        position: fixed !important;
        bottom: 100px !important;
        right: 30px !important;
        width: 400px !important;
        height: 500px !important;
        background: white !important;
        border-radius: 15px !important;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15) !important;
        z-index: 999998 !important;
        display: none !important;
        flex-direction: column !important;
        border: 1px solid #e0e0e0 !important;
        overflow: hidden !important;
        animation: slideUp 0.3s ease-out !important;
    }

    .floating-popup.show {
        display: flex !important;
    }

    @keyframes slideUp {
        from {
            opacity: 0 !important;
            transform: translateY(20px) !important;
        }
        to {
            opacity: 1 !important;
            transform: translateY(0) !important;
        }
    }

    /* Popup header */
    .popup-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        padding: 15px 20px !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        font-weight: bold !important;
    }

    .popup-close {
        background: none !important;
        border: none !important;
        color: white !important;
        font-size: 20px !important;
        cursor: pointer !important;
        padding: 0 !important;
        width: 24px !important;
        height: 24px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 50% !important;
        transition: background-color 0.2s !important;
    }

    .popup-close:hover {
        background-color: rgba(255, 255, 255, 0.2) !important;
    }

    /* Popup content */
    .popup-content {
        flex: 1 !important;
        padding: 20px !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 15px !important;
        background: #fafafa !important;
    }

    /* Notification dot */
    .notification-dot {
        position: absolute !important;
        top: 8px !important;
        right: 8px !important;
        width: 12px !important;
        height: 12px !important;
        background: #ff4757 !important;
        border-radius: 50% !important;
        border: 2px solid white !important;
        animation: pulse 2s infinite !important;
    }

    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(255, 71, 87, 0.7) !important;
        }
        70% {
            box-shadow: 0 0 0 10px rgba(255, 71, 87, 0) !important;
        }
        100% {
            box-shadow: 0 0 0 0 rgba(255, 71, 87, 0) !important;
        }
    }

    /* Responsive design */
    @media (max-width: 480px) {
        .floating-popup {
            width: 320px !important;
            height: 450px !important;
            right: 15px !important;
            bottom: 80px !important;
        }

        .floating-orb-container {
            bottom: 15px !important;
            right: 15px !important;
        }

        .floating-orb {
            width: 50px !important;
            height: 50px !important;
        }

        .orb-icon {
            font-size: 20px !important;
        }
    }
    </style>
    """

    # JavaScript for functionality
    floating_orb_script = """
    <script>
    // Ensure the orb is created only once
    if (!window.floatingOrbCreated) {
        window.floatingOrbCreated = true;

        function createFloatingOrb() {
            // Remove any existing orb
            const existingOrb = document.querySelector('.floating-orb-container');
            if (existingOrb) {
                existingOrb.remove();
            }

            // Create the HTML structure
            const orbHTML = `
                <div class="floating-orb-container">
                    <button class="floating-orb" onclick="window.togglePopup()" title="AI 助手">
                        <span class="orb-icon">🤖</span>
                        <div class="notification-dot"></div>
                    </button>

                    <div class="floating-popup" id="floating-popup">
                        <div class="popup-header">
                            <span>AI 助手</span>
                            <button class="popup-close" onclick="window.closePopup()" title="关闭">&times;</button>
                        </div>
                        <div class="popup-content">
                            <div style="text-align: center; color: #666; margin-top: 50px;">
                                <div style="font-size: 48px; margin-bottom: 20px;">🤖</div>
                                <h3 style="margin: 0 0 10px 0; color: #333;">AI 助手准备就绪</h3>
                                <p style="margin: 0; font-size: 14px;">AI 聊天功能即将上线...</p>
                                <div style="margin-top: 30px; padding: 15px; background: #e8f4fd; border-radius: 8px; font-size: 13px; color: #1976d2;">
                                    💡 即将为您提供智能分析和建议
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Insert into body
            document.body.insertAdjacentHTML('beforeend', orbHTML);

            // Define global functions
            window.togglePopup = function() {
                const popup = document.getElementById('floating-popup');
                const notificationDot = document.querySelector('.notification-dot');

                popup.classList.toggle('show');

                // Hide notification dot when opened
                if (popup.classList.contains('show') && notificationDot) {
                    notificationDot.style.display = 'none';
                }
            };

            window.closePopup = function() {
                const popup = document.getElementById('floating-popup');
                popup.classList.remove('show');
            };

            // Close popup when clicking outside
            document.addEventListener('click', function(e) {
                const container = document.querySelector('.floating-orb-container');
                const popup = document.getElementById('floating-popup');

                if (container && !container.contains(e.target)) {
                    popup.classList.remove('show');
                }
            });

            // Prevent popup from closing when clicking inside it
            const popup = document.getElementById('floating-popup');
            if (popup) {
                popup.addEventListener('click', function(e) {
                    e.stopPropagation();
                });
            }
        }

        // Create orb when document is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', createFloatingOrb);
        } else {
            createFloatingOrb();
        }

        // Monitor for Streamlit reruns and recreate orb if needed
        const observer = new MutationObserver(function() {
            if (!document.querySelector('.floating-orb-container')) {
                setTimeout(createFloatingOrb, 500);
            }
        });

        observer.observe(document.body, { childList: true, subtree: true });
    }
    </script>
    """

    # Inject CSS and JavaScript
    st.markdown(floating_orb_style, unsafe_allow_html=True)
    st.markdown(floating_orb_script, unsafe_allow_html=True)


def show_floating_assistant_status():
    """Show status information about the floating assistant"""
    st.info("🤖 AI助手悬浮球已激活 - 您可以在页面右下角看到悬浮的AI助手球，点击即可打开助手窗口")