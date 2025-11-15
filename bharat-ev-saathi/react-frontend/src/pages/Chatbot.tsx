import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, Send, Trash2, Bot, User } from 'lucide-react';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

const Chatbot = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Add welcome message on component mount
  useEffect(() => {
    setMessages([
      {
        id: '1',
        content: "Hello! 👋 I'm your EV expert assistant. I can help you with:\n\n• EV models and recommendations\n• Subsidy information\n• Charging station locations\n• Battery technology\n• Total cost of ownership\n\nHow can I help you today?",
        sender: 'bot',
        timestamp: new Date()
      }
    ]);
  }, []);

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: inputMessage }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response from server');
      }

      const data = await response.json();

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.response,
        sender: 'bot',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (err) {
      setError('Failed to connect to the chatbot. Please make sure the backend server is running on port 8000.');
      console.error('Chat error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = async () => {
    try {
      await fetch('http://localhost:8000/api/chat/clear', {
        method: 'POST',
      });
      
      setMessages([
        {
          id: Date.now().toString(),
          content: "Chat history cleared! How can I help you today?",
          sender: 'bot',
          timestamp: new Date()
        }
      ]);
      setError(null);
    } catch (err) {
      console.error('Error clearing chat:', err);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4 max-w-6xl">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 flex items-center gap-3">
            <MessageCircle className="text-blue-600" size={40} />
            EV Chatbot Assistant
          </h1>
          <p className="text-gray-600 text-lg">
            Get instant answers about electric vehicles, subsidies, and charging infrastructure
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          {/* Chat Header */}
          <div className="bg-linear-to-r from-blue-600 to-purple-600 text-white p-4 flex justify-between items-center">
            <div className="flex items-center gap-3">
              <Bot size={24} />
              <div>
                <h3 className="font-semibold">EV Expert</h3>
                <p className="text-sm text-blue-100">Online • Ready to help</p>
              </div>
            </div>
            <button
              onClick={clearChat}
              className="p-2 hover:bg-white/20 rounded-lg transition-colors"
              title="Clear chat history"
            >
              <Trash2 size={20} />
            </button>
          </div>

          {/* Messages Area */}
          <div className="h-[500px] overflow-y-auto p-6 bg-gray-50 space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {message.sender === 'bot' && (
                  <div className="shrink-0 w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                    <Bot size={18} className="text-white" />
                  </div>
                )}
                
                <div
                  className={`max-w-[70%] rounded-2xl px-4 py-3 ${
                    message.sender === 'user'
                      ? 'bg-blue-600 text-white rounded-tr-none'
                      : 'bg-white text-gray-800 rounded-tl-none shadow-md'
                  }`}
                >
                  <div className="whitespace-pre-wrap wrap-break-word leading-relaxed">
                    {message.content.split('\n').map((line, idx) => {
                      // Bold headers (lines with **)
                      if (line.includes('**')) {
                        const parts = line.split('**');
                        return (
                          <p key={idx} className="mb-2">
                            {parts.map((part, i) => 
                              i % 2 === 1 ? <strong key={i}>{part}</strong> : part
                            )}
                          </p>
                        );
                      }
                      // Bullet points
                      if (line.trim().startsWith('-') || line.trim().startsWith('•')) {
                        return (
                          <li key={idx} className="ml-4 mb-1">
                            {line.replace(/^[-•]\s*/, '')}
                          </li>
                        );
                      }
                      // Regular lines
                      return line.trim() ? <p key={idx} className="mb-2">{line}</p> : <br key={idx} />;
                    })}
                  </div>
                  <p
                    className={`text-xs mt-2 ${
                      message.sender === 'user' ? 'text-blue-100' : 'text-gray-500'
                    }`}
                  >
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </p>
                </div>

                {message.sender === 'user' && (
                  <div className="shrink-0 w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center">
                    <User size={18} className="text-white" />
                  </div>
                )}
              </div>
            ))}

            {isLoading && (
              <div className="flex gap-3 justify-start">
                <div className="shrink-0 w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                  <Bot size={18} className="text-white" />
                </div>
                <div className="bg-white rounded-2xl rounded-tl-none px-4 py-3 shadow-md">
                  <div className="flex gap-2">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
                <p className="font-semibold">Error</p>
                <p className="text-sm">{error}</p>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="bg-white border-t border-gray-200 p-4">
            <div className="flex gap-3">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about EVs, subsidies, charging stations..."
                className="flex-1 resize-none border border-gray-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows={1}
                disabled={isLoading}
              />
              <button
                onClick={sendMessage}
                disabled={!inputMessage.trim() || isLoading}
                className="bg-blue-600 text-white px-6 py-3 rounded-xl hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2 font-semibold"
              >
                <Send size={20} />
                Send
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Press Enter to send • Shift + Enter for new line
            </p>
          </div>
        </div>

        {/* Features */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-blue-50 rounded-xl p-6">
            <h3 className="font-semibold text-lg mb-2">🚗 EV Recommendations</h3>
            <p className="text-gray-600 text-sm">
              Get personalized EV suggestions based on your budget and requirements
            </p>
          </div>
          <div className="bg-green-50 rounded-xl p-6">
            <h3 className="font-semibold text-lg mb-2">💰 Subsidy Information</h3>
            <p className="text-gray-600 text-sm">
              Learn about FAME-II and state subsidies to save on your EV purchase
            </p>
          </div>
          <div className="bg-purple-50 rounded-xl p-6">
            <h3 className="font-semibold text-lg mb-2">⚡ Charging Stations</h3>
            <p className="text-gray-600 text-sm">
              Find nearby charging stations and learn about charging infrastructure
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
