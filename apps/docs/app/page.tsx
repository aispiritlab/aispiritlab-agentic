import Link from 'next/link';

const cards = [
  {
    title: 'What problem it solves',
    body: 'The workshops turn abstract agent-system ideas into runnable labs, so you can learn tool calling, workflows, runtime orchestration, and distributed coordination step by step.',
  },
  {
    title: 'How it works',
    body: 'Each lab adds one concrete capability: tools, events, planners, runtime buses, multimodal flow, and finally a Redis-backed planner-search-summary pipeline.',
  },
  {
    title: 'How to use it',
    body: 'Install the Python workspace, launch a lab with one command, then use Lab 6 when you want the full distributed demo with Redis, separate services, and external search.',
  },
];

export default function HomePage() {
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col px-6 py-10 sm:px-10 lg:px-12">
      <div className="home-card overflow-hidden rounded-[2rem]">
        <section className="border-b border-slate-200/70 px-8 py-14 sm:px-12">
          <p className="mb-3 text-sm font-semibold uppercase tracking-[0.24em] text-amber-700">
            Workshop-first documentation
          </p>
          <h1 className="max-w-4xl text-5xl leading-tight font-semibold text-slate-950 sm:text-6xl">
            Learn the AI Spirit agent stack by running the labs in order.
          </h1>
          <p className="mt-6 max-w-3xl text-lg leading-8 text-slate-700">
            These docs stay intentionally short: what the tool solves, how each workshop works,
            how to run it, and the full configuration surface for the workshop runtime.
          </p>
          <div className="mt-8 flex flex-wrap gap-4">
            <Link
              className="rounded-full bg-slate-950 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-800"
              href="/docs"
            >
              Open docs
            </Link>
            <Link
              className="rounded-full border border-slate-300 bg-white px-5 py-3 text-sm font-semibold text-slate-800 transition hover:border-slate-400 hover:bg-slate-50"
              href="/docs/reference/configuration"
            >
              Configuration reference
            </Link>
          </div>
        </section>

        <section className="grid gap-5 px-8 py-8 sm:px-12 lg:grid-cols-3">
          {cards.map((card) => (
            <article
              key={card.title}
              className="rounded-[1.5rem] border border-slate-200 bg-white/80 p-6"
            >
              <h2 className="text-2xl font-semibold text-slate-950">{card.title}</h2>
              <p className="mt-3 text-base leading-7 text-slate-700">{card.body}</p>
            </article>
          ))}
        </section>
      </div>
    </main>
  );
}
