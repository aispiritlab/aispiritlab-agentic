import './globals.css';

import type { Metadata } from 'next';
import type { ReactNode } from 'react';

import { Provider } from '@/components/provider';

const metadataBase = process.env.DOCS_SITE_URL
  ? new URL(process.env.DOCS_SITE_URL)
  : undefined;

export const metadata: Metadata = {
  title: {
    default: 'AI Spirit Workshops Docs',
    template: '%s | AI Spirit Workshops Docs',
  },
  description:
    'Workshop-first documentation for the AI Spirit agent labs, from simple tool use to a distributed Redis Streams runtime.',
  metadataBase,
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="docs-shell flex min-h-screen flex-col">
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
