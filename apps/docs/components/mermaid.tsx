'use client';

import { useEffect, useId, useRef, useState } from 'react';
import mermaid from 'mermaid';

mermaid.initialize({
  startOnLoad: false,
  theme: 'default',
  fontFamily: 'var(--font-sans)',
  themeVariables: {
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
  },
});

export function Mermaid({ chart }: { chart: string }) {
  const id = useId().replace(/:/g, '_');
  const containerRef = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState('');

  useEffect(() => {
    let cancelled = false;

    mermaid
      .render(`mermaid-${id}`, chart)
      .then(({ svg: rendered }) => {
        if (!cancelled) setSvg(rendered);
      })
      .catch(console.error);

    return () => {
      cancelled = true;
    };
  }, [chart, id]);

  return (
    <div
      ref={containerRef}
      className="my-6 flex justify-center [&_svg]:max-w-full"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
