import { Shield, History, Settings } from "lucide-react";
import { NavLink } from "@/components/NavLink";
import { useLocation } from "react-router-dom";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar } from
"@/components/ui/sidebar";

const items = [
{ title: "Dashboard", url: "/", icon: Shield },
{ title: "History", url: "/history", icon: History },
{ title: "Settings", url: "/settings", icon: Settings }];


export function AppSidebar() {
  const { state } = useSidebar();
  const collapsed = state === "collapsed";
  const location = useLocation();

  return (
    <Sidebar collapsible="icon" className="border-r border-border">
      <SidebarContent className="pt-4">
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              {items.map((item) =>
              <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <NavLink
                    to={item.url}
                    end={item.url === "/"}
                    className="hover:bg-muted/50 font-mono flex items-center"
                    activeClassName="bg-muted text-primary font-medium">
                      <item.icon className={`w-5 h-5 flex-shrink-0 ${collapsed ? "" : "mr-2"}`} />
                      {!collapsed && <span>{item.title}</span>}
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              )}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>);

}
