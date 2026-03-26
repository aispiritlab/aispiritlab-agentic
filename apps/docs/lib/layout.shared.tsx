import Link from 'next/link';

import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <Link className="font-semibold tracking-tight" href="/docs">
          AI Spirit Workshops
        </Link>
      ),
    },
  };
}
