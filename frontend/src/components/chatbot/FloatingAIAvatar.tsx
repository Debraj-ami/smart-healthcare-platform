import React, { useState } from 'react';
import { Bot, Sparkles } from 'lucide-react';
import { useChatAssistant } from '../../context/ChatAssistantContext';
import { ChatAssistantContextData } from '../../types/chatbot';
import { formatMetricName } from '../../utils/formatters';

interface FloatingAIAvatarProps {
  context?: ChatAssistantContextData;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  showTooltip?: boolean;
  tooltipText?: string;
  position?: 'fixed' | 'inline';
}

export const FloatingAIAvatar: React.FC<FloatingAIAvatarProps> = ({
  context,
  className = '',
  size = 'md',
  showTooltip = true,
  tooltipText,
  position = 'fixed',
}) => {
  const { openChat, pageContext } = useChatAssistant();
  const [isHovered, setIsHovered] = useState(false);
  const [isFocused, setIsFocused] = useState(false);

  const effectiveContext = context || pageContext;

  // Determine dynamic accessible tooltip text
  const resolvedTooltip =
    tooltipText ||
    (effectiveContext?.source === 'assessment' || effectiveContext?.assessmentId
      ? `Ask Smart Health AI about Heart Assessment #${effectiveContext.assessmentId || ''}`
      : effectiveContext?.source === 'monitoring' && effectiveContext?.metric
      ? `Ask Smart Health AI about ${formatMetricName(effectiveContext.metric)} monitoring`
      : 'Open Smart Health AI Assistant');

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    openChat(context);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      e.stopPropagation();
      openChat(context);
    }
  };

  // Dimensions based on size prop
  const sizeClasses = {
    sm: 'w-10 h-10',
    md: 'w-13 h-13 sm:w-14 sm:h-14',
    lg: 'w-16 h-16',
  }[size];

  const iconSizes = {
    sm: 'w-5 h-5',
    md: 'w-6 h-6 sm:w-7 sm:h-7',
    lg: 'w-8 h-8',
  }[size];

  const posClasses =
    position === 'fixed'
      ? 'fixed bottom-6 right-6 z-40'
      : 'relative inline-flex';

  return (
    <div
      className={`${posClasses} items-center justify-center ${className}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Accessible Tooltip */}
      {showTooltip && (isHovered || isFocused) && (
        <div
          role="tooltip"
          id="ai-avatar-tooltip"
          className="absolute bottom-full right-0 mb-3 px-3 py-1.5 bg-slate-900/95 text-white text-xs font-medium rounded-xl shadow-xl whitespace-nowrap pointer-events-none transition-all duration-150 animate-in fade-in slide-in-from-bottom-1 border border-slate-700/50 flex items-center gap-1.5"
        >
          <Sparkles className="w-3.5 h-3.5 text-teal-400" />
          <span>{resolvedTooltip}</span>
          <div className="absolute top-full right-5 -mt-1 border-4 border-transparent border-t-slate-900/95" />
        </div>
      )}

      {/* Circular Avatar Launcher Button */}
      <button
        type="button"
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
        aria-label={resolvedTooltip}
        aria-describedby={showTooltip && (isHovered || isFocused) ? 'ai-avatar-tooltip' : undefined}
        className={`group relative ${sizeClasses} rounded-full bg-gradient-to-tr from-teal-600 via-teal-500 to-cyan-500 text-white shadow-lg hover:shadow-teal-500/25 hover:shadow-xl hover:scale-105 active:scale-95 focus:outline-none focus:ring-4 focus:ring-teal-500/40 focus:ring-offset-2 transition-all duration-200 flex items-center justify-center border-2 border-white/80`}
      >
        {/* Subtle Brand Ring Accent */}
        <span className="absolute -inset-0.5 rounded-full bg-gradient-to-tr from-teal-400 to-cyan-300 opacity-0 group-hover:opacity-40 transition-opacity blur-xs" />

        {/* Smart Health AI Avatar Icon */}
        <div className="relative flex items-center justify-center">
          <Bot className={`${iconSizes} text-white drop-shadow-sm`} />
          <Sparkles className="w-3 h-3 text-cyan-200 absolute -top-1 -right-1 animate-pulse" />
        </div>

        {/* Context Active Notification Dot */}
        {effectiveContext && (
          <span
            className="absolute top-0 right-0 w-3.5 h-3.5 bg-emerald-400 border-2 border-white rounded-full shadow-sm"
            title="Active clinical context attached"
          />
        )}
      </button>
    </div>
  );
};

export default FloatingAIAvatar;
