import HeroSection from "@/components/hero_section";
import HowToUse from "@/components/how_to_use";
import PredictionApp from "@/components/prediction_app";
import Image from "next/image";

export default function Home() {
  return (
    <main>
      <HeroSection />
      <PredictionApp />
      <HowToUse />
    </main>
  );
}
