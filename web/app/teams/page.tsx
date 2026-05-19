import { AppNav } from "../components/AppNav";
import { TeamDashboard } from "../components/TeamDashboard";

export default function TeamsPage() {
  return (
    <main className="app-shell teams-shell">
      <AppNav section="teams" />
      <TeamDashboard />
    </main>
  );
}
