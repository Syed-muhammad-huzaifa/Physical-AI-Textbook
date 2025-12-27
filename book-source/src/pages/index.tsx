import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

const modules = [
  {
    title: 'MODULE 1',
    subtitle: 'ROS2 Architecture',
    description: 'Master ROS2 fundamentals, DDS communication, node patterns, and Python control agents for humanoid robots.',
    icon: '⚙️',
    link: '/docs/MODULE-1-ROS2/ROS2-ARCHITECTURE',
    color: '#22D3EE',
  },
  {
    title: 'MODULE 2',
    subtitle: 'Digital Twin',
    description: 'Build physics-based simulations with Gazebo, design URDF models, and create sensor simulations for testing.',
    icon: '🔄',
    link: '/docs/MODULE-2-DIGITAL-TWIN/GAZEBO-ENVIRONMENT',
    color: '#8B5CF6',
  },
  {
    title: 'MODULE 3',
    subtitle: 'Isaac Sim Platform',
    description: 'Advanced perception, VSLAM localization, synthetic data generation, and sim2real transfer learning.',
    icon: '🤖',
    link: '/docs/MODULE-3-ISAAC/ISAAC-SIM-PLATFORM',
    color: '#10B981',
  },
  {
    title: 'MODULE 4',
    subtitle: 'Vision-Language-Action',
    description: 'End-to-end VLA models, voice interfaces, LLM task planning, and autonomous humanoid capstones.',
    icon: '🧠',
    link: '/docs/MODULE-4-VLA/VISION-LANGUAGE-ACTION',
    color: '#F59E0B',
  },
];

function HeroSection() {
  return (
    <section className={styles.hero}>
      <div className={styles.heroBackground} />
      <div className={styles.heroContent}>
        <div className={styles.heroText}>
          <Heading as="h1" className={styles.heroTitle}>
            Humanoid Robotics Textbook
          </Heading>
          <p className={styles.heroSubtitle}>
            Comprehensive Guide to Humanoid Robotics with ROS2 and AI
          </p>
          <p className={styles.heroDescription}>
            A comprehensive textbook covering ROS2, digital twins, Isaac Sim, and Vision-Language-Action models for humanoid robotics. Build, simulate, and deploy intelligent humanoid robots.
          </p>
          <div className={styles.heroButtons}>
            <Link
              className={styles.primaryButton}
              to="/docs/intro">
              Start Reading
            </Link>
            <Link
              className={styles.secondaryButton}
              to="https://github.com/Syed-muhammad-huzaifa/Physical-AI-Textbook">
              View on GitHub
            </Link>
          </div>
        </div>
        <div className={styles.heroStats}>
          <div className={styles.statCard}>
            <span className={styles.statNumber}>16</span>
            <span className={styles.statLabel}>Chapters</span>
          </div>
          <div className={styles.statCard}>
            <span className={styles.statNumber}>4</span>
            <span className={styles.statLabel}>Modules</span>
          </div>
          <div className={styles.statCard}>
            <span className={styles.statNumber}>2</span>
            <span className={styles.statLabel}>Languages</span>
          </div>
        </div>
      </div>
    </section>
  );
}

function ModuleCard({module}: {module: typeof modules[0]}) {
  return (
    <div className={styles.moduleCard} style={{'--accent-color': module.color} as React.CSSProperties}>
      <div className={styles.moduleIcon}>{module.icon}</div>
      <div className={styles.moduleHeader}>
        <span className={styles.moduleTitle}>{module.title}</span>
        <Heading as="h3" className={styles.moduleSubtitle}>{module.subtitle}</Heading>
      </div>
      <p className={styles.moduleDescription}>{module.description}</p>
      <Link to={module.link} className={styles.moduleLink}>
        Explore Module →
      </Link>
    </div>
  );
}

function ModulesSection() {
  return (
    <section className={styles.modulesSection}>
      <div className={styles.sectionHeader}>
        <Heading as="h2" className={styles.sectionTitle}>Course Modules</Heading>
        <p className={styles.sectionSubtitle}>
          Progress from ROS2 fundamentals to advanced VLA models
        </p>
      </div>
      <div className={styles.modulesGrid}>
        {modules.map((module, idx) => (
          <ModuleCard key={idx} module={module} />
        ))}
      </div>
    </section>
  );
}

function FeaturesSection() {
  const features = [
    {
      title: 'ROS2 Integration',
      description: 'Complete coverage of ROS2 Jazzy/Humble with DDS, services, actions, and lifecycle nodes.',
    },
    {
      title: 'Realistic Simulation',
      description: 'Gazebo and Isaac Sim environments with physics engines and sensor models.',
    },
    {
      title: 'AI-Powered Robotics',
      description: 'VLA models, vision-language models, and LLM-based task planning for humanoids.',
    },
    {
      title: 'Bilingual Support',
      description: 'Full English and Urdu (اردو) translations with RTL support.',
    },
  ];

  return (
    <section className={styles.featuresSection}>
      <div className={styles.sectionHeader}>
        <Heading as="h2" className={styles.sectionTitle}>Key Features</Heading>
      </div>
      <div className={styles.featuresGrid}>
        {features.map((feature, idx) => (
          <div key={idx} className={styles.featureCard}>
            <Heading as="h3" className={styles.featureTitle}>{feature.title}</Heading>
            <p className={styles.featureDescription}>{feature.description}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout
      title="Humanoid Robotics Textbook"
      description="Comprehensive Guide to Humanoid Robotics with ROS2 and AI">
      <main className={styles.main}>
        <HeroSection />
        <ModulesSection />
        <FeaturesSection />
      </main>
    </Layout>
  );
}
