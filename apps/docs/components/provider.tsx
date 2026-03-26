'use client';

import { RootProvider } from 'fumadocs-ui/provider/next';
import type { ReactNode } from 'react';

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
const searchFrom = `${basePath}/api/search`;

export function Provider({ children }: { children: ReactNode }) {
  return (
    <RootProvider
      search={{
        options: {
          api: searchFrom,
          type: 'static',
        },
      }}
    >
      {children}
    </RootProvider>
  );
}
