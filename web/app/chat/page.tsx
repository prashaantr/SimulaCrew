import { AppNav } from "../components/AppNav";
import { ChatWorkspace } from "../components/ChatWorkspace";

export default function ChatPage() {
  return (
    <main className="app-shell chat-shell">
      <AppNav section="chat" />
      <ChatWorkspace />
    </main>
  );
}
