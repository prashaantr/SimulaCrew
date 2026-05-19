import Link from "next/link";

type AppNavProps = {
  section: "teams" | "chat" | "runs";
};

export function AppNav({ section }: AppNavProps) {
  return (
    <header className="app-nav">
      <Link className="brand-lockup" href="/teams">
        <span className="brand-mark">SC</span>
        <span>
          <strong>SimulaCrew</strong>
          <small>modern agent terminal</small>
        </span>
      </Link>
      <nav aria-label="Primary">
        <Link className={section === "teams" ? "active" : ""} href="/teams">
          /teams
        </Link>
        <Link className={section === "chat" ? "active" : ""} href="/chat">
          /chat
        </Link>
        <Link className={section === "runs" ? "active" : ""} href="/runs">
          /runs
        </Link>
      </nav>
    </header>
  );
}
