'use client';

import { useEffect, useId, useState } from 'react';
import mermaid from 'mermaid';
import { useTheme } from 'next-themes';

const lightThemeVars = {
  primaryColor: '#dbeafe',
  primaryTextColor: '#1e293b',
  primaryBorderColor: '#3b82f6',
  lineColor: '#64748b',
  secondaryColor: '#fef3c7',
  tertiaryColor: '#f1f5f9',
  noteBkgColor: '#fefce8',
  noteTextColor: '#1e293b',
  actorTextColor: '#1e293b',
  actorBkg: '#eff6ff',
  actorBorder: '#3b82f6',
  signalColor: '#1e293b',
  signalTextColor: '#1e293b',
  labelTextColor: '#1e293b',
  loopTextColor: '#1e293b',
  activationBorderColor: '#3b82f6',
  sequenceNumberColor: '#ffffff',
};

const darkThemeVars = {
  primaryColor: '#1e3a5f',
  primaryTextColor: '#e2e8f0',
  primaryBorderColor: '#60a5fa',
  lineColor: '#94a3b8',
  secondaryColor: '#422006',
  tertiaryColor: '#1e293b',
  noteBkgColor: '#1c1917',
  noteTextColor: '#e2e8f0',
  actorTextColor: '#e2e8f0',
  actorBkg: '#1e293b',
  actorBorder: '#60a5fa',
  signalColor: '#e2e8f0',
  signalTextColor: '#e2e8f0',
  labelTextColor: '#e2e8f0',
  loopTextColor: '#e2e8f0',
  activationBorderColor: '#60a5fa',
  sequenceNumberColor: '#ffffff',
};

export function Mermaid({ chart }: { chart: string }) {
  const id = useId().replace(/:/g, '_');
  const [svg, setSvg] = useState('');
  const { resolvedTheme } = useTheme();

  useEffect(() => {
    const isDark = resolvedTheme === 'dark';
    let cancelled = false;

    mermaid.initialize({
      startOnLoad: false,
      theme: 'base',
      fontFamily: 'var(--font-sans)',
      themeVariables: isDark ? darkThemeVars : lightThemeVars,
    });

    mermaid
      .render(`mermaid-${id}-${isDark ? 'd' : 'l'}`, chart)
      .then(({ svg: rendered }) => {
        if (!cancelled) setSvg(rendered);
      })
      .catch(console.error);

    return () => {
      cancelled = true;
    };
  }, [chart, id, resolvedTheme]);

  return (
    <div
      className="my-6 flex justify-center [&_svg]:max-w-full"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
