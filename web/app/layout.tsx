import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SimulaCrew Chat",
  description: "Group-chat interface for SimulaCrew agent simulations"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
