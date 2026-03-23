import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import ThreatHeader from "@/components/ThreatHeader";
import type { ReactNode } from "react";

interface AppLayoutProps {
  children: ReactNode;
}

const AppLayout = ({ children }: AppLayoutProps) => {
  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full bg-background grid-bg relative">
        <div className="fixed inset-0 scanline-overlay z-0" />
        <AppSidebar />
        <div className="flex-1 flex flex-col relative z-10">
          <ThreatHeader />
          <div className="flex items-center border-b border-border px-4 h-10">
            <SidebarTrigger className="text-muted-foreground" />
          </div>
          <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6">
            {children}
          </main>
        </div>
      </div>
    </SidebarProvider>
  );
};

export default AppLayout;
