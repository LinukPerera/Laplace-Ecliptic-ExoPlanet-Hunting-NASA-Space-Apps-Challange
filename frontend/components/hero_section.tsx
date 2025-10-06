"use client";
import { DotLottieReact } from "@lottiefiles/dotlottie-react";
import Galaxy from "./galaxy";
import { useEffect, useRef } from "react";
import Link from "next/link";

function animateImageRef(titleRef: React.RefObject<any>) {
    if (!titleRef.current) return;
    titleRef.current.style.webkitMaskBoxImage = 'linear-gradient(to right, transparent 50%, black 50%, transparent 50%)';
    titleRef.current.style.maskImage = 'linear-gradient(to right, transparent 50%, black 50%, transparent 50%)';
    let pos1 = 50;
    let pos2 = 50;
    
    const animate = () =>{
        if (!titleRef.current) return;
        pos1 -= 1;
        pos2 += 1;
        if (pos1 <= -50) {
            return;
        }
        if (pos2 >= 150) {
            return;
        }
        titleRef.current.style.webkitMaskBoxImage = `linear-gradient(to right, transparent ${pos1}%, black 50%, transparent ${pos2}%)`;
        titleRef.current.style.maskImage = `linear-gradient(to right, transparent ${pos1}%, black 50%, transparent ${pos2}%)`;
        requestAnimationFrame(animate);
    }

    requestAnimationFrame(animate);
}

export default function HeroSection() {
    const titleRef = useRef<HTMLHeadingElement>(null);
    const subtitleRef = useRef<HTMLParagraphElement>(null);

    useEffect(() => {
        if (titleRef.current) {
            setTimeout(() => {
                titleRef.current!.classList.remove('hidden');
                titleRef.current!.classList.add('animate-fade-in');
                animateImageRef(titleRef);

            }, 600);

            setTimeout(() => {
                subtitleRef.current!.classList.remove('hidden');
                subtitleRef.current!.classList.add('animate-fade-in');
            }, 800);


            // scale on Scroll down
            const handleScroll = () => {
                const scrollY = window.scrollY;
                const scale = Math.max(1 - scrollY / 1000, 0.5);
                titleRef.current!.style.transform = `scale(${scale})`;
                titleRef.current!.style.opacity = `${scale}`;
                subtitleRef.current!.style.transform = `scale(${scale})`;
                subtitleRef.current!.style.opacity = `${scale}`;
            }
            window.addEventListener('scroll', handleScroll);

            return () => {
                window.removeEventListener('scroll', handleScroll);
            }
        }
    }, []);

    return (
        
        <div className="h-screen w-full bg-black gradient-bg-1 flex items-center justify-center relative overflow-hidden">
            {/*<DotLottieReact src="/anims/Space Areal.lottie" loop autoplay className="h-full w-full object-cover opacity-20" />*/}
            <header className="absolute w-full top-0 left-0 z-20 flex items-center justify-center py-2 animate-fade-in-down">
                <Link href="/" className="p-4 text-white text-lg font-medium hover:text-yellow-300 transition-colors">Home</Link>
                <Link href="/#app" className="p-4 text-white text-lg font-medium hover:text-yellow-300 transition-colors">App</Link>
                <Link href="/#how-to-use" className="p-4 text-white text-lg font-medium hover:text-yellow-300 transition-colors">How to use</Link>
            </header>
            <Galaxy />
            <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-4">
                
                <h1 className="text-8xl max-md:text-6xl font-bold text-white mb-4 uppercase orbitron-font hidden" ref={titleRef}>Laplace Ecleptic</h1>
                <p className="text-2xl font-mono font-bold text-white hidden" ref={subtitleRef}>Predict Exo-Planets from Raw Data</p>
            </div>
        </div>
    )
}