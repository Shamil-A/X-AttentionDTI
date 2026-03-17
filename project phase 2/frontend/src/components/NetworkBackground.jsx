import { useCallback } from "react";
import Particles from "react-tsparticles";
import { loadSlim } from "tsparticles-slim";

export default function NetworkBackground() {
  const particlesInit = useCallback(async (engine) => {
    await loadSlim(engine);
  }, []);

  return (
    <Particles
      id="tsparticles"
      init={particlesInit}
      options={{
        fullScreen: {
          enable: true,
          zIndex: -1 
        },
        background: {
          color: { value: "#08090f" }, // Matches your exact --color-brand-bg
        },
        fpsLimit: 60,
        interactivity: {
          events: {
            onHover: {
              enable: true,
              mode: "grab",
            },
          },
          modes: {
            grab: {
              distance: 140,
              links: { opacity: 0.6, color: "#38d9f5" }, // Grabs light up in Cyan!
            },
          },
        },
        particles: {
          color: { value: "#7c6cf4" }, // Particles are now your brand Purple
          links: {
            color: "#6b7280", // Links are subtle muted gray
            distance: 150,
            enable: true,
            opacity: 0.3,
            width: 1.2,
          },
          move: {
            direction: "none",
            enable: true,
            outModes: { default: "bounce" },
            random: false,
            speed: 0.8,
            straight: false,
          },
          number: {
            density: { enable: true, area: 800 },
            value: 60,
          },
          opacity: { value: 0.6 },
          shape: { type: "circle" },
          size: { value: { min: 1, max: 3 } },
        },
        detectRetina: true,
      }}
    />
  );
}