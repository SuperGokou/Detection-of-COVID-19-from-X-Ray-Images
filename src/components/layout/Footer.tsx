import { AlertTriangle } from "lucide-react";

export function Footer() {
  return (
    <footer className="bg-white border-t border-[#e5e7eb] mt-auto">
      <div className="px-8 py-8 flex items-center justify-between">
        <div className="flex flex-col gap-1">
          <p className="text-sm text-[#6a7282]">
            &copy; 2024 Harvard University &amp; Class COMPSCI 1090B.
          </p>
          <p className="text-sm text-[#6a7282]">
            Version 1.0.0
          </p>
        </div>
        <div className="flex items-center gap-6">
          <span className="flex items-center gap-2 text-sm text-[#6a7282]">
            <AlertTriangle className="size-4 text-amber-500" />
            Research Use Only
          </span>
          <a href="#" className="text-sm text-[#6a7282] hover:text-[#364153] transition-colors">
            Privacy Policy
          </a>
          <a href="#" className="text-sm text-[#6a7282] hover:text-[#364153] transition-colors">
            Terms of Service
          </a>
        </div>
      </div>
    </footer>
  );
}
