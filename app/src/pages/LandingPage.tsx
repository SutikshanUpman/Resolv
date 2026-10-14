import { PageMeta } from '../components/layout/PageMeta';
import { Hero } from '../components/landing/Hero';
import { HowItWorks } from '../components/landing/HowItWorks';
import { KeyIdeas } from '../components/landing/KeyIdeas';
import { SameBusiness } from '../components/landing/SameBusiness';
import { StatBand } from '../components/landing/StatBand';
import { TeamSection } from '../components/landing/TeamSection';

export function LandingPage() {
  return (
    <>
      <PageMeta />
      <Hero />
      <StatBand />
      <SameBusiness />
      <HowItWorks />
      <KeyIdeas />
      <TeamSection />
    </>
  );
}
