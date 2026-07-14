"use client";

import React, { useEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import {
  Phone,
  MousePointerClick,
  LayoutDashboard,
  Calendar,
  Users,
  Stethoscope,
  Pill,
  FileText,
  Search,
  Bell,
  ChevronDown,
  TrendingUp,
  Activity,
  BarChart3,
  Network,
  BedDouble,
  Database,
  Quote,
  ChevronLeft,
  ChevronRight,
  Smartphone
} from "lucide-react";

export default function LandingPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const mockupRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    gsap.registerPlugin(ScrollTrigger);
    
    const ctx = gsap.context(() => {
      // Navbar animation
      gsap.from(".nav-bar", {
        y: -20,
        opacity: 0,
        duration: 0.8,
        ease: "power3.out",
        delay: 0.1
      });

      // Hero text stagger
      gsap.from(".hero-item", {
        y: 30,
        opacity: 0,
        duration: 0.8,
        stagger: 0.1,
        ease: "power3.out",
        delay: 0.2
      });

      // Mockup entrance
      gsap.from(mockupRef.current, {
        y: 60,
        opacity: 0,
        duration: 1,
        ease: "power3.out",
        delay: 0.4
      });

      // Fade up elements on scroll
      gsap.utils.toArray<HTMLElement>(".fade-up").forEach(elem => {
        gsap.from(elem, {
          scrollTrigger: {
            trigger: elem,
            start: "top 85%",
          },
          y: 40,
          opacity: 0,
          duration: 0.8,
          ease: "power3.out"
        });
      });
    }, containerRef);

    return () => ctx.revert();
  }, []);

  return (
    <div className="min-h-screen bg-black font-sans text-[#F5F5F5] selection:bg-[#14967f]/30" style={{ fontFamily: "'Inter', sans-serif" }} ref={containerRef}>
      
      {/* Outer container: reduced max-width and padding for tighter layout */}
      <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-col gap-24">
        
        {/* HERO SECTION */}
        <section className="relative">
          <div className="bg-gradient-to-b from-zinc-900 to-black border border-[#F5F5F5]/5 rounded-[3rem] pt-6 pb-20 px-4 sm:px-8 relative overflow-hidden">
            
            {/* Minimal Navbar */}
            <nav className="nav-bar relative z-20 flex items-center justify-between max-w-[1000px] mx-auto bg-black rounded-full p-2 shadow-xl border border-[#F5F5F5]/10 backdrop-blur-md">
              <div className="flex items-center pl-4 pr-2 shrink-0">
                <span className="text-[15px] font-bold tracking-wide text-[#14967f]">Nirikshon</span>
              </div>
              
              <div className="hidden lg:flex items-center gap-8 px-4">
                <a href="#" className="text-[13px] font-bold text-[#F5F5F5]/80 hover:text-[#F5F5F5] transition-colors">Home</a>
                <a href="#" className="text-[13px] font-bold text-[#F5F5F5]/80 hover:text-[#F5F5F5] transition-colors">Work</a>
                <a href="#" className="text-[13px] font-bold text-[#F5F5F5]/80 hover:text-[#F5F5F5] transition-colors">Prices</a>
                <a href="#" className="text-[13px] font-bold text-[#F5F5F5]/80 hover:text-[#F5F5F5] transition-colors">Projects</a>
                <a href="#" className="text-[13px] font-bold text-[#F5F5F5]/80 hover:text-[#F5F5F5] transition-colors">Contact</a>
              </div>

              <button className="hidden md:flex items-center justify-center h-10 px-6 rounded-full bg-[#14967f] hover:bg-[#14967f] transition-all text-[13px] font-bold text-[#095d7e] shrink-0">
                Get Started
              </button>
            </nav>

            {/* Hero Content: Tighter spacing, smaller typography */}
            <div className="relative z-10 flex flex-col items-center justify-center mt-20 text-center max-w-3xl mx-auto">
              <h1 className="hero-item text-5xl sm:text-[56px] font-extrabold tracking-tight text-[#F5F5F5] leading-[1.05]">
                Smart healthcare <br /> decisions start here
              </h1>
              <p className="hero-item mt-6 text-[15px] sm:text-base text-[#E0E0E0] max-w-[500px] mx-auto leading-relaxed font-medium">
                An intuitive dashboard that turns complex data into clear insights — for hospitals, clinics, and care teams.
              </p>
              
              <div className="hero-item flex flex-col sm:flex-row items-center gap-3 mt-8">
                <button className="flex items-center gap-2 px-6 py-3 rounded-full bg-[#a8e6cf] text-[#095d7e] hover:bg-[#bcf2dc] transition-all text-[14px] font-bold shadow-lg shadow-[#14967f]/10 hover:-translate-y-0.5">
                  <MousePointerClick className="w-4 h-4" />
                  Try demo
                </button>
                <button className="flex items-center gap-2 px-6 py-3 rounded-full border border-[#F5F5F5]/10 bg-[#F5F5F5]/5 hover:bg-[#F5F5F5]/10 transition-all text-[14px] font-bold text-[#F5F5F5] shadow-sm backdrop-blur-sm hover:-translate-y-0.5">
                  <Phone className="w-4 h-4" />
                  Book a call
                </button>
              </div>
            </div>

          {/* Embedded Dashboard Mockup - Max width 900px, tighter paddings */}
          <div className="relative z-30 mx-auto mt-20 w-full max-w-[1200px] px-4" ref={mockupRef}>
            <div className="w-full bg-zinc-950 rounded-3xl p-2 shadow-2xl shadow-emerald-900/10 border border-[#F5F5F5]/10 ring-1 ring-white/5 flex">
              {/* Sidebar */}
              <div className="w-56 bg-zinc-900 rounded-2xl p-5 flex-col gap-6 hidden lg:flex border border-[#F5F5F5]/5">
                <div className="flex items-center gap-2.5">
                  <div className="w-9 h-9 rounded-xl bg-[#14967f] text-[#095d7e] flex items-center justify-center font-bold text-[15px] shadow-inner shadow-[#a8e6cf]/20">SM</div>
                  <div className="leading-tight">
                    <div className="font-bold text-[#F5F5F5] text-[14px]">St. Mark</div>
                    <div className="text-[11px] text-[#E0E0E0] font-medium">hospital</div>
                  </div>
                </div>

                <div className="flex flex-col gap-1 mt-2">
                  <a href="#" className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl bg-[#14967f] text-[#095d7e] font-semibold text-[13px] shadow-sm shadow-[#a8e6cf]/20">
                    <LayoutDashboard className="w-3.5 h-3.5" /> Dashboard
                  </a>
                  <a href="#" className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-[#E0E0E0] hover:bg-[#F5F5F5]/5 font-medium text-[13px] transition-colors">
                    <Calendar className="w-3.5 h-3.5" /> Schedule
                  </a>
                  <a href="#" className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-[#E0E0E0] hover:bg-[#F5F5F5]/5 font-medium text-[13px] transition-colors">
                    <Users className="w-3.5 h-3.5" /> Patients
                  </a>
                  <a href="#" className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-[#E0E0E0] hover:bg-[#F5F5F5]/5 font-medium text-[13px] transition-colors">
                    <Stethoscope className="w-3.5 h-3.5" /> Doctors
                  </a>
                  <a href="#" className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-[#E0E0E0] hover:bg-[#F5F5F5]/5 font-medium text-[13px] transition-colors">
                    <Pill className="w-3.5 h-3.5" /> Medications
                  </a>
                  <a href="#" className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-[#E0E0E0] hover:bg-[#F5F5F5]/5 font-medium text-[13px] transition-colors">
                    <FileText className="w-3.5 h-3.5" /> Reports
                  </a>
                </div>
              </div>

              {/* Main Content */}
              <div className="flex-1 p-6 flex flex-col gap-6 bg-zinc-900 rounded-r-[1.5rem]">
                {/* Topbar */}
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-extrabold text-[#F5F5F5] tracking-tight">Overview</h2>
                  <div className="flex items-center gap-4">
                    <div className="relative hidden md:block">
                      <Search className="w-3.5 h-3.5 absolute left-3.5 top-1/2 -translate-y-1/2 text-zinc-500" />
                      <input type="text" placeholder="Search" className="pl-9 pr-3 py-1.5 rounded-full bg-zinc-800 border border-[#F5F5F5]/5 focus:bg-zinc-950 focus:border-[#14967f] focus:ring-4 focus:ring-[#14967f]/20 outline-none text-[13px] text-[#F5F5F5] w-60 transition-all font-medium" />
                    </div>
                    <button className="w-8 h-8 rounded-full bg-zinc-800 border border-[#F5F5F5]/5 flex items-center justify-center text-[#E0E0E0] hover:bg-zinc-700 transition-colors">
                      <Bell className="w-4 h-4" />
                    </button>
                    <div className="flex items-center gap-2.5 pl-2.5 border-l border-[#F5F5F5]/10 cursor-pointer">
                      <div className="w-8 h-8 rounded-full bg-[#14967f]/20 flex items-center justify-center text-[#14967f] font-bold text-[10px] uppercase">MA</div>
                      <div className="hidden lg:block leading-tight">
                        <div className="text-[13px] font-bold text-[#F5F5F5]">Maria</div>
                        <div className="text-[9px] text-[#E0E0E0] font-bold tracking-wider uppercase">Administrator</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Grid Content */}
                <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
                  {/* Visitor Statistic Card */}
                  <div className="xl:col-span-2 bg-zinc-950 border border-[#F5F5F5]/5 rounded-2xl p-5 shadow-sm flex flex-col">
                    <div className="flex justify-between items-start mb-6">
                      <div>
                        <h3 className="font-bold text-[15px] text-[#F5F5F5]">Visitor Statistic</h3>
                        <p className="text-[12px] text-[#E0E0E0] font-medium mt-0.5">Total amount of visitors and patients</p>
                      </div>
                      <select className="text-[12px] border border-[#F5F5F5]/10 rounded-md px-2 py-1 bg-zinc-900 outline-none text-zinc-300 font-semibold cursor-pointer">
                        <option>For year</option>
                        <option>For month</option>
                      </select>
                    </div>
                    
                    {/* Fake Chart */}
                    <div className="flex-1 flex items-end justify-between gap-3 h-40 relative">
                      <div className="absolute left-0 top-0 h-full w-full flex flex-col justify-between z-0 pb-5">
                        <div className="border-t border-[#F5F5F5]/5 border-dashed w-full"></div>
                        <div className="border-t border-[#F5F5F5]/5 border-dashed w-full"></div>
                        <div className="border-t border-[#F5F5F5]/5 border-dashed w-full"></div>
                        <div className="border-t border-[#F5F5F5]/5 border-dashed w-full"></div>
                      </div>
                      
                      {[40, 65, 45, 80, 55, 75].map((val, i) => (
                        <div key={i} className="flex flex-col items-center gap-2 z-10 w-full h-full justify-end">
                          <div className="w-full flex gap-1 justify-center items-end h-[85%]">
                            <div className="w-1/2 max-w-[16px] bg-zinc-800 border border-[#F5F5F5]/5 rounded-t-sm relative group overflow-hidden">
                              <div className="absolute bottom-0 w-full bg-zinc-600 rounded-t-sm transition-all duration-500" style={{ height: `${val}%` }}></div>
                            </div>
                            <div className="w-1/2 max-w-[16px] bg-[#095d7e]/30 border border-emerald-500/20 rounded-t-sm relative group overflow-hidden">
                              <div className="absolute bottom-0 w-full bg-[#14967f] rounded-t-sm transition-all duration-500" style={{ height: `${val * 0.6}%` }}></div>
                            </div>
                          </div>
                          <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">
                            {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'][i]}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Right Cards Column */}
                  <div className="flex flex-col gap-5">
                    {/* Total Patients Card */}
                    <div className="bg-zinc-950 border border-[#F5F5F5]/5 rounded-2xl p-5 text-[#F5F5F5] flex flex-col justify-between h-[130px] relative overflow-hidden group">
                      <div className="flex justify-between items-start relative z-10">
                        <div>
                          <h3 className="font-semibold text-[#F5F5F5] text-[13px]">Total Patients</h3>
                          <p className="text-[10px] text-[#E0E0E0] mt-0.5">including robust visits</p>
                        </div>
                        <div className="bg-[#14967f]/20 text-[#14967f] backdrop-blur-md px-2 py-0.5 rounded-full text-[10px] font-bold flex items-center gap-1">
                          3.5k <TrendingUp className="w-3 h-3" />
                        </div>
                      </div>
                      <div className="relative z-10 mt-2">
                        <div className="text-3xl font-extrabold tracking-tight">521</div>
                        <div className="text-[10px] text-[#E0E0E0] mt-0.5">Since last month</div>
                      </div>
                      <div className="absolute bottom-0 right-0 p-3 flex items-end gap-1 opacity-40 z-0">
                        {[30, 40, 25, 50, 40, 60, 80, 55, 70, 90].map((h, i) => (
                          <div key={i} className="w-[3px] bg-[#14967f]/50 rounded-t-sm" style={{ height: `${h * 0.8}px` }}></div>
                        ))}
                      </div>
                    </div>

                    {/* Blood Donations Card */}
                    <div className="bg-zinc-950 border border-[#F5F5F5]/5 rounded-2xl p-5 text-[#F5F5F5] flex flex-col justify-between h-[130px] relative group">
                      <div className="flex justify-between items-start relative z-10">
                        <div>
                          <h3 className="font-semibold text-[#F5F5F5] text-[13px]">Blood Donations</h3>
                          <p className="text-[10px] text-[#E0E0E0] mt-0.5">volunteer donation program</p>
                        </div>
                        <div className="bg-[#14967f]/20 px-2 py-0.5 rounded-full text-[10px] font-bold text-[#a8e6cf] flex items-center gap-1">
                          2.5k <TrendingUp className="w-3 h-3" />
                        </div>
                      </div>
                      <div className="relative z-10 mt-2">
                        <div className="text-3xl font-extrabold tracking-tight text-[#F5F5F5]">355</div>
                        <div className="text-[10px] text-[#E0E0E0] mt-0.5">Since last month</div>
                      </div>
                      <div className="absolute bottom-3 right-3 flex items-center gap-1 opacity-80 z-0">
                         <svg width="80" height="35" viewBox="0 0 100 40" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-zinc-600">
                           <path d="M0 25 L20 15 L40 35 L60 20 L80 30 L100 10" />
                           <circle cx="20" cy="15" r="4" fill="#09090b" stroke="#10b981" strokeWidth="2" />
                           <circle cx="40" cy="35" r="4" fill="#09090b" stroke="#10b981" strokeWidth="2" />
                           <circle cx="60" cy="20" r="4" fill="#09090b" stroke="#10b981" strokeWidth="2" />
                           <circle cx="80" cy="30" r="4" fill="#09090b" stroke="#10b981" strokeWidth="2" />
                         </svg>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          </div>
          
        </section>

        {/* TRUSTED BY LOGOS */}
        <section className="fade-up max-w-[1200px] mx-auto w-full pt-10 pb-8 mb-10">
          <p className="text-center text-[12px] font-bold text-slate-400 mb-8 uppercase tracking-widest">Trusted by leading healthcare teams</p>
          <div className="flex flex-wrap justify-center items-center gap-10 md:gap-16 opacity-40 hover:opacity-100 transition-all duration-700">
            <div className="font-extrabold text-xl tracking-tighter flex items-center gap-2 text-[#F5F5F5]"><Activity className="w-5 h-5"/> MEDI-PRO</div>
            <div className="font-extrabold text-xl tracking-tighter flex items-center gap-2 text-[#F5F5F5]"><Stethoscope className="w-5 h-5"/> CLINIC<span className="font-light">CARE</span></div>
            <div className="font-extrabold text-xl tracking-tighter flex items-center gap-2 text-[#F5F5F5]"><Database className="w-5 h-5"/> DATACROSS</div>
            <div className="font-extrabold text-xl tracking-tighter flex items-center gap-2 text-[#F5F5F5]"><Network className="w-5 h-5"/> VITALNET</div>
          </div>
        </section>

        {/* FEATURES SECTION */}
        <section className="fade-up max-w-[1200px] mx-auto w-full pt-6">
          <div className="text-center mb-10">
            <h2 className="text-[28px] md:text-3xl font-extrabold text-[#F5F5F5] tracking-tight">Built for modern healthcare</h2>
          </div>
          
          <div className="w-full h-[320px] bg-zinc-900 rounded-3xl overflow-hidden relative shadow-sm group">
            <img 
              src="https://images.unsplash.com/photo-1581594693702-fbdc51b2763b?auto=format&fit=crop&w=2000&q=80" 
              alt="Modern Healthcare Facility" 
              className="w-full h-full object-cover opacity-90 transition-transform duration-1000 group-hover:scale-[1.03]"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-10 mt-12">
            <FeatureItem 
              icon={<Activity />}
              title="Real-time patient monitoring"
              desc="Track the number of visitors, hospitalizations, and consultations in real-time."
            />
            <FeatureItem 
              icon={<BarChart3 />}
              title="Reports by day, week, month"
              desc="Generate reports on patient visits, payment sources, patient categories, and more."
            />
            <FeatureItem 
              icon={<Network />}
              title="Department workload analytics"
              desc="Analyze which specialties are most in demand and which require optimization."
            />
            <FeatureItem 
              icon={<Search />}
              title="Integration with internal systems"
              desc="Compatible with your existing EHRs, registries, and medical databases."
            />
            <FeatureItem 
              icon={<BedDouble />}
              title="Convenient room occupancy table"
              desc="Easily check which doctors are available, when, and where – without confusion."
            />
            <FeatureItem 
              icon={<Database />}
              title="Centralized patient database"
              desc="Quick access to medical records, treatment history, and doctor profiles."
            />
          </div>
        </section>

        {/* SECTION: Understand your patients */}
        <section className="fade-up max-w-[1200px] mx-auto w-full flex flex-col md:flex-row items-center gap-12 py-6">
          <div className="flex-1 space-y-6">
            <h2 className="text-[32px] md:text-[36px] font-extrabold text-[#F5F5F5] tracking-tight leading-[1.1]">
              Understand your <br className="hidden md:block"/> patients at a glance
            </h2>
            <p className="text-[15px] text-[#E0E0E0] font-medium leading-relaxed max-w-sm">
              Explore the new dashboard showing user trends by department and financing type, helping you plan resources effectively.
            </p>
            <button className="flex items-center gap-2 px-6 py-3 rounded-full bg-[#a8e6cf] text-[#095d7e] hover:bg-[#bcf2dc] transition-all text-[14px] font-bold shadow-md shadow-[#a8e6cf]/20 w-fit">
              <MousePointerClick className="w-4 h-4" />
              Try demo
            </button>
          </div>
          <div className="flex-1 w-full bg-zinc-900/50 border border-[#F5F5F5]/5 rounded-3xl p-8 relative overflow-hidden flex items-center justify-center min-h-[300px]">
            <div className="bg-zinc-950 rounded-2xl shadow-lg p-5 w-full max-w-[320px] border border-[#F5F5F5]/10 relative z-10">
               <div className="flex justify-between items-center mb-5">
                 <div>
                   <h4 className="font-bold text-[#F5F5F5] text-[13px]">Visitor Statistic</h4>
                   <div className="w-16 h-1.5 bg-zinc-800 rounded-full mt-1.5"></div>
                 </div>
                 <div className="w-12 h-5 bg-zinc-800 rounded-md"></div>
               </div>
               <div className="flex items-end justify-between h-24 gap-2">
                  {[40, 70, 45, 90, 60].map((v, i) => (
                    <div key={i} className="flex gap-[2px] items-end h-full w-full justify-center">
                       <div className="w-full max-w-[8px] bg-zinc-900 rounded-t-sm"><div className="bg-zinc-700 rounded-t-sm w-full" style={{height: `${v}%`}}></div></div>
                       <div className="w-full max-w-[8px] bg-[#095d7e]/30 rounded-t-sm"><div className="bg-[#14967f] rounded-t-sm w-full" style={{height: `${v*0.6}%`}}></div></div>
                    </div>
                  ))}
               </div>
            </div>
            
            <div className="absolute top-6 right-6 bg-zinc-900 rounded-xl p-4 shadow-lg text-[#F5F5F5] w-40 transform rotate-6 z-20 border border-[#F5F5F5]/10">
              <div className="text-[10px] font-semibold text-[#E0E0E0] uppercase tracking-wide">Total Patients</div>
              <div className="text-xl font-extrabold mt-0.5">521</div>
              <div className="mt-3 flex gap-1 h-5 items-end opacity-60">
                {[30,40,20,50,40,60,80].map((h, i) => <div key={i} className="w-[3px] bg-white rounded-t-sm" style={{height: `${h}%`}}></div>)}
              </div>
            </div>
          </div>
        </section>

        {/* SECTION: Always within reach */}
        <section className="fade-up max-w-[1200px] mx-auto w-full flex flex-col-reverse md:flex-row items-center gap-12 py-6">
          <div className="flex-1 w-full bg-zinc-900/50 border border-[#F5F5F5]/5 rounded-3xl p-8 flex justify-center items-center min-h-[350px] relative overflow-hidden">
             
             <div className="relative w-full max-w-[280px] h-full flex justify-center items-center">
                {/* Phone 1 */}
                <div className="absolute left-0 z-10 w-40 bg-zinc-950 rounded-[1.5rem] shadow-xl border-[5px] border-zinc-800 h-[280px] overflow-hidden transform -rotate-6">
                   <div className="w-1/3 h-3 bg-black absolute top-0 left-1/2 -translate-x-1/2 rounded-b-lg z-20"></div>
                   <div className="p-3 pt-6 bg-zinc-900 h-full flex flex-col gap-3">
                     <div className="flex justify-between items-center"><div className="w-12 h-3 bg-zinc-800 rounded-md"></div><div className="w-5 h-5 rounded-full bg-zinc-800"></div></div>
                     <div className="w-full h-16 bg-zinc-950 rounded-xl shadow-sm border border-[#F5F5F5]/5 p-2"><div className="w-10 h-2 bg-zinc-800 rounded mb-2"></div><div className="flex gap-1.5 h-6 items-end mt-2"><div className="w-2 h-full bg-[#14967f] rounded-t-[1px]"></div><div className="w-2 h-2/3 bg-[#14967f]/30 rounded-t-[1px]"></div></div></div>
                     <div className="w-full h-10 bg-zinc-950 rounded-xl shadow-sm border border-[#F5F5F5]/5 flex items-center px-2 gap-2"><div className="w-4 h-4 rounded-full bg-zinc-800"></div><div className="w-12 h-1.5 bg-zinc-800 rounded"></div></div>
                     <div className="w-full h-10 bg-zinc-950 rounded-xl shadow-sm border border-[#F5F5F5]/5 flex items-center px-2 gap-2"><div className="w-4 h-4 rounded-full bg-zinc-800"></div><div className="w-12 h-1.5 bg-zinc-800 rounded"></div></div>
                   </div>
                </div>
                {/* Phone 2 */}
                <div className="absolute right-0 top-8 z-20 w-40 bg-zinc-900 rounded-[1.5rem] shadow-2xl shadow-[#14967f]/10 border-[5px] border-zinc-700 h-[280px] overflow-hidden transform rotate-3">
                   <div className="w-1/3 h-3 bg-zinc-950 absolute top-0 left-1/2 -translate-x-1/2 rounded-b-lg z-20"></div>
                   <div className="p-3 pt-6 bg-zinc-900 h-full flex flex-col gap-3">
                     <div className="w-full h-24 bg-zinc-800 border border-[#F5F5F5]/5 rounded-xl p-3"><div className="w-12 h-2 bg-zinc-700 rounded mb-3"></div><div className="text-[#F5F5F5] font-extrabold text-xl">355</div></div>
                     <div className="w-full h-12 bg-zinc-800/50 rounded-xl p-2"><div className="w-8 h-1.5 bg-zinc-700 rounded mb-1.5"></div><div className="w-16 h-1.5 bg-zinc-700 rounded"></div></div>
                     <div className="w-full h-12 bg-zinc-800/50 rounded-xl p-2"><div className="w-8 h-1.5 bg-zinc-700 rounded mb-1.5"></div><div className="w-16 h-1.5 bg-zinc-700 rounded"></div></div>
                   </div>
                </div>
             </div>

          </div>
          <div className="flex-1 space-y-6">
            <h2 className="text-[32px] md:text-[36px] font-extrabold text-[#F5F5F5] tracking-tight leading-[1.1]">
              Always within reach
            </h2>
            <p className="text-[15px] text-[#E0E0E0] font-medium leading-relaxed max-w-sm">
              Stay connected on the go with our mobile-friendly dashboard. Access key stats and patient data anywhere – right from your smartphone.
            </p>
            <button className="flex items-center gap-2 px-6 py-3 rounded-full bg-[#a8e6cf] text-[#095d7e] hover:bg-[#bcf2dc] transition-all text-[14px] font-bold shadow-md shadow-[#a8e6cf]/20 w-fit">
              <Smartphone className="w-4 h-4" />
              Try demo
            </button>
          </div>
        </section>

        {/* WORK / PROJECTS (CASE STUDIES) */}
        <section className="fade-up max-w-[1200px] mx-auto w-full py-16">
          <div className="text-center mb-12">
            <h2 className="text-[28px] md:text-3xl font-extrabold text-[#F5F5F5] tracking-tight">Our Work & Impact</h2>
            <p className="text-[15px] text-[#E0E0E0] font-medium max-w-lg mx-auto mt-3">See how leading hospitals transformed their workflows with Nirikshon.</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Project 1 */}
            <div className="group rounded-[2rem] bg-zinc-950 border border-[#F5F5F5]/5 p-2 overflow-hidden hover:shadow-xl hover:shadow-[#14967f]/10 transition-all duration-500">
               <div className="w-full h-64 bg-blue-100 rounded-[1.5rem] overflow-hidden relative">
                  <img src="https://images.unsplash.com/photo-1519494026892-80bbd2d6fd0d?auto=format&fit=crop&w=1000&q=80" className="absolute inset-0 w-full h-full object-cover" alt="Hospital Room" />
                  <div className="absolute inset-0 bg-slate-900/60 group-hover:bg-slate-900/40 transition-colors"></div>
                  <div className="absolute inset-0 p-6 flex flex-col justify-end text-[#F5F5F5] z-10">
                    <span className="text-[11px] font-bold uppercase tracking-wider mb-1 opacity-80">St. Mark's Hospital</span>
                    <h3 className="text-xl font-extrabold leading-tight">Reduced wait times by 40%</h3>
                  </div>
               </div>
               <div className="p-6">
                 <p className="text-[14px] text-[#E0E0E0] font-medium">Implemented the predictive flow dashboard to manage emergency room capacity in real-time.</p>
                 <a href="#" className="inline-flex items-center gap-1.5 text-[13px] font-bold text-[#14967f] mt-4 group-hover:text-[#a8e6cf]">Read case study <ChevronRight className="w-3.5 h-3.5" /></a>
               </div>
            </div>
            {/* Project 2 */}
            <div className="group rounded-[2rem] bg-zinc-950 border border-[#F5F5F5]/5 p-2 overflow-hidden hover:shadow-xl hover:shadow-[#14967f]/10 transition-all duration-500">
               <div className="w-full h-64 bg-[#e0f7fa] rounded-[1.5rem] overflow-hidden relative">
                  <img src="https://images.unsplash.com/photo-1538108149393-cebb47ac52b0?auto=format&fit=crop&w=1000&q=80" className="absolute inset-0 w-full h-full object-cover" alt="Medical Clinic" />
                  <div className="absolute inset-0 bg-slate-900/60 group-hover:bg-slate-900/40 transition-colors"></div>
                  <div className="absolute inset-0 p-6 flex flex-col justify-end text-[#F5F5F5] z-10">
                    <span className="text-[11px] font-bold uppercase tracking-wider mb-1 opacity-80">Mercy Clinic Network</span>
                    <h3 className="text-xl font-extrabold leading-tight">Unified 12 regional databases</h3>
                  </div>
               </div>
               <div className="p-6">
                 <p className="text-[14px] text-[#E0E0E0] font-medium">Integrated fragmented EHRs into a single, lightning-fast dashboard for 500+ daily users.</p>
                 <a href="#" className="inline-flex items-center gap-1.5 text-[13px] font-bold text-[#14967f] mt-4 group-hover:text-[#a8e6cf]">Read case study <ChevronRight className="w-3.5 h-3.5" /></a>
               </div>
            </div>
          </div>
        </section>

        {/* PRICING SECTION */}
        <section className="fade-up max-w-[1200px] mx-auto w-full py-16">
          <div className="text-center mb-12">
            <h2 className="text-[28px] md:text-3xl font-extrabold text-[#F5F5F5] tracking-tight">Simple, transparent prices</h2>
            <p className="text-[15px] text-[#E0E0E0] font-medium max-w-lg mx-auto mt-3">Choose the plan that fits your facility's scale.</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
            {/* Starter */}
            <div className="rounded-[2rem] bg-zinc-950 border border-[#F5F5F5]/5 p-8 hover:border-[#F5F5F5]/20 transition-all">
              <h3 className="text-[15px] font-bold text-[#F5F5F5]">Starter</h3>
              <div className="mt-4 mb-6"><span className="text-4xl font-extrabold text-[#F5F5F5]">$99</span><span className="text-zinc-500 font-medium text-[13px]">/mo</span></div>
              <ul className="space-y-4 mb-8">
                {['Up to 5 users', 'Basic analytics', 'Standard support'].map((item, i) => (
                  <li key={i} className="flex items-center gap-3 text-[14px] text-[#E0E0E0] font-medium">
                    <svg className="w-4 h-4 text-[#14967f]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 6L9 17l-5-5"/></svg>
                    {item}
                  </li>
                ))}
              </ul>
              <button className="w-full py-3 rounded-full border border-[#F5F5F5]/10 text-[13px] font-bold text-[#F5F5F5] hover:bg-[#F5F5F5]/5 transition-colors">Start free trial</button>
            </div>
            
            {/* Pro */}
            <div className="rounded-[2rem] bg-[#14967f] text-[#095d7e] p-8 shadow-xl shadow-[#a8e6cf]/20 md:-my-4 relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-[#b2ebf2] to-[#80deea]"></div>
              <div className="inline-block px-3 py-1 bg-[#ffd54f] rounded-full text-[10px] font-bold text-[#095d7e] mb-4 tracking-wider uppercase">Most Popular</div>
              <h3 className="text-[15px] font-bold text-[#095d7e]">Professional</h3>
              <div className="mt-4 mb-6"><span className="text-4xl font-extrabold text-[#095d7e]">$299</span><span className="text-[#095d7e]/80 font-medium text-[13px]">/mo</span></div>
              <ul className="space-y-4 mb-8">
                {['Unlimited users', 'Advanced predictive AI', 'EHR integrations', '24/7 priority support'].map((item, i) => (
                  <li key={i} className="flex items-center gap-3 text-[14px] text-[#095d7e]/80 font-medium">
                    <svg className="w-4 h-4 text-[#095d7e]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 6L9 17l-5-5"/></svg>
                    {item}
                  </li>
                ))}
              </ul>
              <button className="w-full py-3 rounded-full bg-black text-[13px] font-bold text-[#F5F5F5] hover:bg-zinc-900 transition-colors">Get Started</button>
            </div>

            {/* Enterprise */}
            <div className="rounded-[2rem] bg-zinc-950 border border-[#F5F5F5]/5 p-8 hover:border-[#F5F5F5]/20 transition-all">
              <h3 className="text-[15px] font-bold text-[#F5F5F5]">Enterprise</h3>
              <div className="mt-4 mb-6"><span className="text-4xl font-extrabold text-[#F5F5F5]">Custom</span></div>
              <ul className="space-y-4 mb-8">
                {['Custom deployments', 'Dedicated success manager', 'On-premise options'].map((item, i) => (
                  <li key={i} className="flex items-center gap-3 text-[14px] text-[#E0E0E0] font-medium">
                    <svg className="w-4 h-4 text-[#14967f]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 6L9 17l-5-5"/></svg>
                    {item}
                  </li>
                ))}
              </ul>
              <button className="w-full py-3 rounded-full border border-[#F5F5F5]/10 text-[13px] font-bold text-[#F5F5F5] hover:bg-[#F5F5F5]/5 transition-colors">Contact sales</button>
            </div>
          </div>
        </section>

        {/* FAQ SECTION */}
        <section className="fade-up max-w-[1000px] mx-auto w-full py-16">
          <div className="text-center mb-10">
            <h2 className="text-[28px] md:text-3xl font-extrabold text-[#F5F5F5] tracking-tight">Frequently Asked Questions</h2>
          </div>
          <div className="space-y-4">
            {[
              {q: "How long does integration take?", a: "Most clinics are fully integrated within 14 days, with our team handling 90% of the heavy lifting."},
              {q: "Is patient data secure?", a: "Yes. Nirikshon is HIPAA compliant and uses enterprise-grade AES-256 encryption for all data at rest and in transit."},
              {q: "Do you offer training for staff?", a: "Absolutely. All plans include onboarding sessions, and Professional/Enterprise plans include ongoing training."},
              {q: "Can it connect with legacy EHRs?", a: "We support integrations with most major EHRs (Epic, Cerner, etc.) via HL7 and modern FHIR APIs."}
            ].map((faq, i) => (
              <div key={i} className="border border-[#F5F5F5]/10 rounded-[1.5rem] p-5 hover:border-[#F5F5F5]/30 transition-colors cursor-pointer group bg-zinc-950">
                <div className="flex justify-between items-center">
                  <h4 className="font-bold text-[15px] text-[#F5F5F5]">{faq.q}</h4>
                  <ChevronDown className="w-4 h-4 text-zinc-500 group-hover:text-[#F5F5F5] transition-colors" />
                </div>
                {/* Note: In a real app this would be state-driven, but we'll leave it simple for visual design */}
                <p className="text-[14px] text-[#E0E0E0] font-medium mt-3 hidden group-hover:block">{faq.a}</p>
              </div>
            ))}
          </div>
        </section>

        {/* TESTIMONIALS */}
        <section className="fade-up max-w-[1200px] mx-auto w-full pt-6">
          <div className="text-center mb-10 space-y-3">
            <h2 className="text-[28px] md:text-[32px] font-extrabold text-[#F5F5F5] tracking-tight">Customers thoughts</h2>
            <p className="text-[#E0E0E0] font-medium max-w-lg mx-auto text-[14px] leading-relaxed">We value every insight, from our users - here's what doctors and clinic staff say about our platform.</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            <TestimonialCard 
              quote="Managing resources and space is now faster and clearer. The system speaks to the art of minimalism."
              name="Cody Fisher"
              role="Purchasing Coordinator"
            />
            <TestimonialCard 
              quote="This dashboard has completely transformed how we manage patient flow and schedules. Incredibly helpful."
              name="Dr. Anna Lindberg"
              role="Head of Internal Medicine"
              active
            />
            <TestimonialCard 
              quote="Finally, a tool that gives us real-time data we can trust. It helps make better decisions every day."
              name="Johan Pettersson"
              role="Clinic Administrator"
            />
          </div>

          <div className="flex justify-center items-center gap-3 mt-10">
             <button className="w-10 h-10 rounded-full border border-[#F5F5F5]/10 flex items-center justify-center text-zinc-500 hover:bg-zinc-800 hover:text-[#F5F5F5] transition-colors bg-zinc-950">
               <ChevronLeft className="w-4 h-4" />
             </button>
             <button className="w-10 h-10 rounded-full border border-[#F5F5F5]/10 flex items-center justify-center text-[#F5F5F5] hover:bg-zinc-800 transition-colors bg-zinc-900 shadow-sm">
               <ChevronRight className="w-4 h-4" />
             </button>
          </div>
        </section>

        {/* CONTACT SECTION */}
        <section className="fade-up max-w-[800px] mx-auto w-full py-16">
          <div className="bg-zinc-950 rounded-[2rem] border border-[#F5F5F5]/5 p-8 md:p-12 shadow-sm relative overflow-hidden">
             <div className="absolute top-0 right-0 w-32 h-32 bg-[#095d7e]/10 rounded-bl-[100px] -z-0"></div>
             <div className="relative z-10 text-center mb-8">
               <h2 className="text-[28px] font-extrabold text-[#F5F5F5] tracking-tight">Get in touch</h2>
               <p className="text-[14px] text-[#E0E0E0] font-medium mt-2">Have a specific question? Send us a message.</p>
             </div>
             <form className="relative z-10 space-y-4">
               <div className="grid grid-cols-2 gap-4">
                 <div className="space-y-1.5">
                   <label className="text-[12px] font-bold text-[#E0E0E0] px-1">First name</label>
                   <input type="text" className="w-full h-12 bg-zinc-900 border border-[#F5F5F5]/5 rounded-xl px-4 text-[14px] text-[#F5F5F5] font-medium outline-none focus:border-[#14967f] focus:ring-1 focus:ring-[#14967f] transition-all placeholder-zinc-600" placeholder="John" />
                 </div>
                 <div className="space-y-1.5">
                   <label className="text-[12px] font-bold text-[#E0E0E0] px-1">Last name</label>
                   <input type="text" className="w-full h-12 bg-zinc-900 border border-[#F5F5F5]/5 rounded-xl px-4 text-[14px] text-[#F5F5F5] font-medium outline-none focus:border-[#14967f] focus:ring-1 focus:ring-[#14967f] transition-all placeholder-zinc-600" placeholder="Doe" />
                 </div>
               </div>
               <div className="space-y-1.5">
                 <label className="text-[12px] font-bold text-[#E0E0E0] px-1">Email</label>
                 <input type="email" className="w-full h-12 bg-zinc-900 border border-[#F5F5F5]/5 rounded-xl px-4 text-[14px] text-[#F5F5F5] font-medium outline-none focus:border-[#14967f] focus:ring-1 focus:ring-[#14967f] transition-all placeholder-zinc-600" placeholder="john@hospital.com" />
               </div>
               <div className="space-y-1.5">
                 <label className="text-[12px] font-bold text-[#E0E0E0] px-1">Message</label>
                 <textarea className="w-full h-32 bg-zinc-900 border border-[#F5F5F5]/5 rounded-xl p-4 text-[14px] text-[#F5F5F5] font-medium outline-none focus:border-[#14967f] focus:ring-1 focus:ring-[#14967f] transition-all resize-none placeholder-zinc-600" placeholder="How can we help you?"></textarea>
               </div>
               <button type="button" className="w-full h-12 rounded-xl bg-[#a8e6cf] text-[#095d7e] font-bold text-[14px] hover:bg-[#bcf2dc] transition-colors mt-2 shadow-lg shadow-[#a8e6cf]/20">Send message</button>
             </form>
          </div>
        </section>

        {/* BOTTOM CTA */}
        <section className="fade-up max-w-[950px] mx-auto w-full pb-8">
          <div className="bg-gradient-to-b from-zinc-900 to-black border border-[#F5F5F5]/5 rounded-[2.5rem] py-16 px-6 text-center flex flex-col items-center justify-center relative overflow-hidden">
            
            <h2 className="text-[32px] md:text-[40px] font-extrabold text-[#F5F5F5] tracking-tight relative z-10 leading-tight">
              Ready to simplify <br/> your hospital's data?
            </h2>
            <p className="mt-4 text-[15px] text-[#E0E0E0] font-medium max-w-[450px] mx-auto relative z-10 leading-relaxed">
              An intuitive dashboard that turns complex data into clear insights — for hospitals, clinics, and care teams.
            </p>
            
            <div className="flex flex-col sm:flex-row items-center gap-3 mt-8 relative z-10">
              <button className="flex items-center gap-2 px-6 py-3 rounded-full bg-[#a8e6cf] text-[#095d7e] hover:bg-[#bcf2dc] transition-all text-[14px] font-bold shadow-lg shadow-[#a8e6cf]/20 hover:-translate-y-0.5">
                <MousePointerClick className="w-4 h-4" />
                Try demo
              </button>
              <button className="flex items-center gap-2 px-6 py-3 rounded-full border border-[#F5F5F5]/10 bg-zinc-900/70 hover:bg-zinc-800 transition-all text-[14px] font-bold text-[#F5F5F5] shadow-sm backdrop-blur-sm hover:-translate-y-0.5">
                <Phone className="w-4 h-4" />
                Book a call
              </button>
            </div>
          </div>
        </section>

      </div>
      
      {/* FOOTER - Moved out of the max-w container to be full width or just tightly bounded */}
      <div className="max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8 pb-4">
        <footer className="fade-up w-full bg-zinc-950 border border-[#F5F5F5]/5 rounded-[2rem] p-8 sm:p-12 text-[#F5F5F5]">
          <div className="flex flex-col md:flex-row justify-between items-start gap-10">
            <div className="space-y-6 flex-1">
              <h2 className="text-[26px] font-bold tracking-tight">Let's connect!</h2>
              <div>
                <div className="text-[17px] font-bold flex items-center gap-2 mb-1.5">
                  <div className="w-5 h-5 rounded-full bg-[#14967f] flex items-center justify-center">
                    <Activity className="w-3 h-3 text-[#095d7e]" />
                  </div>
                  Nirikshon
                </div>
                <p className="text-zinc-500 text-[13px] font-medium leading-relaxed max-w-[240px]">
                  8502 Preston Rd. Inglewood, Maine 98380<br/>
                  hello@Nirikshon.com<br/>
                  (406) 555-0120
                </p>
              </div>
              <div className="flex gap-3">
                <button className="bg-black rounded-lg px-3 py-1.5 flex items-center gap-2 border border-[#F5F5F5]/10 hover:bg-zinc-900 transition-colors">
                  <svg className="w-4 h-4 text-[#F5F5F5]" viewBox="0 0 24 24" fill="currentColor"><path d="M17.05 20.28c-.98.95-2.05.8-3.08.35-1.09-.46-2.09-.48-3.24 0-1.44.62-2.2.44-3.06-.35C2.79 15.25 3.51 7.59 9.05 7.31c1.35.07 2.29.74 3.08.8 1.18-.09 2.31-.86 3.65-.72 1.46.15 2.6.76 3.34 1.87-3.03 1.83-2.52 6.01.5 7.27-.67 1.63-1.6 3.02-2.57 3.75zM12.03 7.25c-.15-2.23 1.66-4.07 3.74-4.25.29 2.58-2.34 4.5-3.74 4.25z"/></svg>
                  <div className="flex flex-col items-start leading-none">
                    <span className="text-[9px] text-[#E0E0E0]">Download on the</span>
                    <span className="text-[12px] font-semibold">App Store</span>
                  </div>
                </button>
                <button className="bg-black rounded-lg px-3 py-1.5 flex items-center gap-2 border border-[#F5F5F5]/10 hover:bg-zinc-900 transition-colors">
                  <svg className="w-4 h-4 text-[#F5F5F5]" viewBox="0 0 24 24" fill="currentColor"><path d="M3.73 21.65l10.9-10.9L3.73 2.35v19.3zM15.86 9.53l-1.23-1.23L4.93.42C4.5.08 3.99.16 3.73.4l12.13 9.13zM16.94 10.42l3.43-2.58c.84-.63.84-1.65 0-2.28l-3.43-2.58-1.58 1.58 1.58 5.86zM15.86 11.3l-1.23 1.23-9.7 7.88c.43.34.94.26 1.2-.02l9.73-9.09z"/></svg>
                  <div className="flex flex-col items-start leading-none">
                    <span className="text-[9px] text-[#E0E0E0]">GET IT ON</span>
                    <span className="text-[12px] font-semibold">Google Play</span>
                  </div>
                </button>
              </div>
            </div>

            <div className="flex flex-col justify-between items-end gap-10 h-full text-right w-full md:w-auto">
               <button className="flex items-center gap-1.5 px-5 py-2 rounded-full border border-[#F5F5F5]/20 hover:bg-white hover:text-[#095d7e] transition-all text-[13px] font-semibold text-[#F5F5F5]">
                 <Phone className="w-3.5 h-3.5" />
                 Book a call
               </button>
               
               <div className="flex flex-wrap justify-end gap-5 text-[13px] font-semibold text-[#E0E0E0]">
                 <a href="#" className="hover:text-[#F5F5F5] transition-colors">For hospitals</a>
                 <a href="#" className="hover:text-[#F5F5F5] transition-colors">For doctors</a>
                 <a href="#" className="hover:text-[#F5F5F5] transition-colors">About</a>
                 <a href="#" className="hover:text-[#F5F5F5] transition-colors">Technologies</a>
               </div>

               <div className="flex items-center gap-3 mt-auto">
                  <p className="text-[11px] text-zinc-500 font-medium mr-3">© 2026 Nirikshon, All rights reserved.</p>
                  <a href="#" className="w-7 h-7 rounded-full bg-zinc-900 border border-[#F5F5F5]/5 flex items-center justify-center hover:bg-zinc-800 transition-colors"><InstagramIcon className="w-3.5 h-3.5" /></a>
                  <a href="#" className="w-7 h-7 rounded-full bg-zinc-900 border border-[#F5F5F5]/5 flex items-center justify-center hover:bg-zinc-800 transition-colors"><YoutubeIcon className="w-3.5 h-3.5" /></a>
                  <a href="#" className="w-7 h-7 rounded-full bg-zinc-900 border border-[#F5F5F5]/5 flex items-center justify-center hover:bg-zinc-800 transition-colors"><FacebookIcon className="w-3.5 h-3.5" /></a>
                  <a href="#" className="w-7 h-7 rounded-full bg-zinc-900 border border-[#F5F5F5]/5 flex items-center justify-center hover:bg-zinc-800 transition-colors"><TwitterIcon className="w-3.5 h-3.5" /></a>
               </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}

function FeatureItem({ icon, title, desc }: { icon: React.ReactNode, title: string, desc: string }) {
  return (
    <div className="flex flex-col items-start gap-3">
      <div className="w-10 h-10 rounded-xl bg-[#14967f]/10 flex items-center justify-center text-[#14967f]">
        {React.cloneElement(icon as React.ReactElement<any>, { className: "w-5 h-5" })}
      </div>
      <h3 className="font-bold text-[15px] text-[#F5F5F5]">{title}</h3>
      <p className="text-[13px] text-[#E0E0E0] font-medium leading-relaxed">{desc}</p>
    </div>
  );
}

function TestimonialCard({ quote, name, role, active = false }: { quote: string, name: string, role: string, active?: boolean }) {
  return (
    <div className={`rounded-[1.5rem] p-6 flex flex-col gap-4 transition-all duration-300 border ${active ? 'bg-zinc-900 shadow-xl shadow-[#14967f]/10 border-[#F5F5F5]/10 scale-[1.02]' : 'bg-zinc-950 border-transparent hover:bg-zinc-900 hover:shadow-md hover:border-[#F5F5F5]/5'}`}>
       <Quote className={`w-6 h-6 ${active ? 'text-[#14967f]' : 'text-zinc-600'}`} fill="currentColor" />
       <p className={`font-semibold text-[14px] leading-relaxed flex-1 ${active ? 'text-[#F5F5F5]' : 'text-[#E0E0E0]'}`}>"{quote}"</p>
       <div className="flex items-center gap-3">
         <div className="w-9 h-9 rounded-full bg-zinc-800 overflow-hidden">
            <img src={`https://ui-avatars.com/api/?name=${encodeURIComponent(name)}&background=random`} alt={name} className="w-full h-full object-cover" />
         </div>
         <div className="leading-tight">
           <div className="font-bold text-[13px] text-[#F5F5F5]">{name}</div>
           <div className="text-[11px] font-medium text-zinc-500 mt-0.5">{role}</div>
         </div>
       </div>
    </div>
  );
}

function FacebookIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M18 2h-3a5 5 0 0 0-5 5v3H7v4h3v8h4v-8h3l1-4h-4V7a1 1 0 0 1 1-1h3z" />
    </svg>
  );
}

function TwitterIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M22 4s-.7 2.1-2 3.4c1.6 10-9.4 17.3-18 11.6 2.2.1 4.4-.6 6-2C3 15.5.5 9.6 3 5c2.2 2.6 5.6 4.1 9 4-.9-4.2 4-6.6 7-3.8 1.1 0 3-1.2 3-1.2z" />
    </svg>
  );
}

function InstagramIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <rect x="2" y="2" width="20" height="20" rx="5" ry="5" />
      <path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z" />
      <line x1="17.5" y1="6.5" x2="17.51" y2="6.5" />
    </svg>
  );
}

function YoutubeIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M22.54 6.42a2.78 2.78 0 0 0-1.94-2C18.88 4 12 4 12 4s-6.88 0-8.6.46a2.78 2.78 0 0 0-1.94 2A29 29 0 0 0 1 11.75a29 29 0 0 0 .46 5.33A2.78 2.78 0 0 0 3.4 19c1.72.46 8.6.46 8.6.46s6.88 0 8.6-.46a2.78 2.78 0 0 0 1.94-2 29 29 0 0 0 .46-5.25 29 29 0 0 0-.46-5.33z" />
      <polygon points="9.75 15.02 15.5 11.75 9.75 8.48 9.75 15.02" />
    </svg>
  );
}
