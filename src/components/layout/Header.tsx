import { FileText, Download, User } from "lucide-react";

export function Header() {
  return (
    <header className="bg-white border-b border-black/80 px-8 h-16 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-5">
          <div className="size-8 bg-[#a51c30] rounded-md flex items-center justify-center">
            <span className="text-white font-bold text-base font-serif">H</span>
          </div>
          <div className="w-px h-6 bg-[#d1d5dc]" />
          <div className="size-8 bg-[#1c398e] rounded-full flex items-center justify-center">
            <span className="text-white font-bold text-xs">U</span>
          </div>
        </div>
        <h1 className="text-lg font-bold text-[#1e1e1e]">
          COVID-19 Diagnostic Model Sandbox
        </h1>
      </div>

      <div className="flex items-center">
        <button className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-gray-100 transition-colors">
          <FileText className="size-4 text-[#4a5565]" />
          <span className="text-base text-[#4a5565]">Documentation</span>
        </button>
        <button className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-gray-100 transition-colors">
          <Download className="size-4 text-[#4a5565]" />
          <span className="text-base text-[#4a5565]">Export Results</span>
        </button>
        <div className="w-px h-6 bg-[#d1d5dc] mx-2" />
        <button className="size-10 flex items-center justify-center rounded-lg hover:bg-gray-100 transition-colors">
          <User className="size-4 text-[#4a5565]" />
        </button>
      </div>
    </header>
  );
}
