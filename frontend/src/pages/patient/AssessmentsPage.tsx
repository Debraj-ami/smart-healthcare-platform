import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePatient } from '../../context/PatientContext';
import {
  FileCheck2,
  Heart,
  Droplet,
  Brain,
  ShieldAlert,
  Info,
  ChevronRight,
  Clock,
  Sparkles,
  X,
  MessageSquareHeart,
  Activity,
  ClipboardList,
} from 'lucide-react';
import { DiseaseAssessment } from '../../types/assessment';
import assessmentService from '../../services/assessmentService';
import Button from '../../components/common/Button';
import Card from '../../components/common/Card';
import Badge from '../../components/common/Badge';
import { formatDateTime } from '../../utils/formatters';
import LiverDiseaseAssessmentModal from '../../components/assessments/LiverDiseaseAssessmentModal';
import HeartDiseaseAssessmentModal from '../../components/assessments/HeartDiseaseAssessmentModal';
import HeartAssessmentHistory from '../../components/assessments/HeartAssessmentHistory';
import HeartAssessmentDetailModal from '../../components/assessments/HeartAssessmentDetailModal';

export const AssessmentsPage: React.FC = () => {
  const { activePatientId, currentUser } = usePatient();
  const navigate = useNavigate();
  const [assessments, setAssessments] = useState<DiseaseAssessment[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [selectedDisease, setSelectedDisease] = useState<string | null>(null);
  const [showHeartPathwaysModal, setShowHeartPathwaysModal] = useState<boolean>(false);
  const [showClinicalGatewayModal, setShowClinicalGatewayModal] = useState<boolean>(false);
  const [activeHistoryTab, setActiveHistoryTab] = useState<'all' | 'heart' | 'liver'>('all');
  const [selectedHeartAssessment, setSelectedHeartAssessment] = useState<DiseaseAssessment | null>(null);
  const [refreshHistoryCount, setRefreshHistoryCount] = useState<number>(0);

  const fetchAssessments = async () => {
    setIsLoading(true);
    try {
      const data = await assessmentService.getAssessmentsByPatient(activePatientId);
      setAssessments(data);
    } catch (err) {
      console.error('Failed to fetch assessments', err);
      setAssessments([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAssessments();
  }, [activePatientId]);

  const handleAssessmentComplete = () => {
    fetchAssessments();
    setRefreshHistoryCount((prev) => prev + 1);
  };

  const diseaseCards = [
    {
      type: 'DIABETES',
      title: 'Diabetes Risk',
      description: 'Assess early risk of type-2 diabetes using glucose, insulin, and BMI markers.',
      icon: Droplet,
      color: 'bg-teal-50 text-teal-600',
    },
    {
      type: 'HEART_DISEASE',
      title: 'Heart Disease Risk',
      description: 'Evaluate cardiovascular health markers, resting blood pressure, and cholesterol.',
      icon: Heart,
      color: 'bg-rose-50 text-rose-600',
    },
    {
      type: 'PARKINSONS',
      title: 'Parkinson’s Disease Risk',
      description: 'Analyze acoustic vocal frequency variations and motor biomarker characteristics.',
      icon: Brain,
      color: 'bg-purple-50 text-purple-600',
    },
    {
      type: 'LIVER_DISEASE',
      title: 'Liver Disease Risk',
      description: 'Screen for hepatic irregularities through bilirubin, albumin, and enzyme levels.',
      icon: ShieldAlert,
      color: 'bg-amber-50 text-amber-600',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Patient-Centric Primary Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          How can Smart Health help you today?
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-1 max-w-3xl leading-relaxed">
          Talk about symptoms, monitor changes in your health, or use validated clinical results for an AI-assisted screening assessment.
        </p>
      </div>

      {/* Primary Patient Pathways (Symptoms | Monitoring | Clinical Assessment) */}
      <section aria-labelledby="patient-pathways-heading" className="space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
          <h2 id="patient-pathways-heading" className="text-xs font-bold uppercase tracking-wider text-slate-400">
            Care & Health Pathways
          </h2>
          <span className="text-[11px] font-medium text-slate-400">
            Choose the path that matches what you have available today
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {/* Card 1: Symptoms */}
          <div className="bg-white rounded-2xl p-6 border border-slate-200 hover:border-teal-300 shadow-sm hover:shadow-card transition-all flex flex-col justify-between space-y-5 group">
            <div className="space-y-3.5">
              <div className="flex items-center justify-between">
                <div className="w-12 h-12 rounded-xl bg-teal-50 text-teal-600 flex items-center justify-center font-bold group-hover:scale-105 transition-transform">
                  <MessageSquareHeart className="w-6 h-6" />
                </div>
                <Badge variant="teal" size="sm">Conversational</Badge>
              </div>
              <div>
                <h3 className="text-lg font-bold text-slate-900">Talk about symptoms</h3>
                <p className="text-xs sm:text-sm text-slate-600 mt-1.5 leading-relaxed">
                  Discuss what you're experiencing and get safety-focused conversational guidance.
                </p>
              </div>
              <div className="pt-2 border-t border-slate-100 text-[11px] text-slate-500">
                <p className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-teal-500 flex-shrink-0" />
                  <span>Immediate safety triage • No clinical report required</span>
                </p>
              </div>
            </div>
            <div>
              <Button
                variant="teal"
                size="md"
                onClick={() => navigate('/chat')}
                className="w-full text-xs font-bold justify-center shadow-xs"
                aria-label="Talk to Assistant about symptoms"
              >
                Talk to Assistant
              </Button>
            </div>
          </div>

          {/* Card 2: Monitoring */}
          <div className="bg-white rounded-2xl p-6 border border-slate-200 hover:border-blue-300 shadow-sm hover:shadow-card transition-all flex flex-col justify-between space-y-5 group">
            <div className="space-y-3.5">
              <div className="flex items-center justify-between">
                <div className="w-12 h-12 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center font-bold group-hover:scale-105 transition-transform">
                  <Activity className="w-6 h-6" />
                </div>
                <Badge variant="info" size="sm">Tracking & Trends</Badge>
              </div>
              <div>
                <h3 className="text-lg font-bold text-slate-900">Monitor my health</h3>
                <p className="text-xs sm:text-sm text-slate-600 mt-1.5 leading-relaxed">
                  Track available health measurements, personal baselines, and changes over time.
                </p>
              </div>
              <div className="pt-2 border-t border-slate-100 text-[11px] text-slate-500">
                <p className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
                  <span>Track vitals over time • Personal deviation analysis</span>
                </p>
              </div>
            </div>
            <div>
              <Button
                variant="primary"
                size="md"
                onClick={() => navigate('/monitoring')}
                className="w-full text-xs font-bold justify-center shadow-xs"
                aria-label="Open Health Monitoring"
              >
                Open Monitoring
              </Button>
            </div>
          </div>

          {/* Card 3: Clinical Results */}
          <div className="bg-white rounded-2xl p-6 border border-slate-200 hover:border-rose-300 shadow-sm hover:shadow-card transition-all flex flex-col justify-between space-y-5 group">
            <div className="space-y-3.5">
              <div className="flex items-center justify-between">
                <div className="w-12 h-12 rounded-xl bg-rose-50 text-rose-600 flex items-center justify-center font-bold group-hover:scale-105 transition-transform">
                  <ClipboardList className="w-6 h-6" />
                </div>
                <Badge variant="danger" size="sm">Clinical Panel</Badge>
              </div>
              <div>
                <h3 className="text-lg font-bold text-slate-900">I have clinical results</h3>
                <p className="text-xs sm:text-sm text-slate-600 mt-1.5 leading-relaxed">
                  Choose a disease-specific screening module when you have validated clinical or laboratory measurements.
                </p>
              </div>
              <div className="pt-2 border-t border-slate-100 text-[11px] text-slate-500">
                <p className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-rose-500 flex-shrink-0" />
                  <span>This option is for patients who already have the required validated clinical measurements.</span>
                </p>
              </div>
            </div>
            <div>
              <Button
                variant="outline"
                size="md"
                onClick={() => setShowClinicalGatewayModal(true)}
                className="w-full text-xs font-bold justify-center border-rose-200 text-rose-700 hover:bg-rose-50 shadow-xs"
                aria-label="Open Clinical Assessment Gateway"
              >
                Clinical Assessment
              </Button>
            </div>
          </div>
        </div>

        <p className="text-[11px] text-slate-400 pt-1">
          * Non-diagnostic screening notice: Smart Health provides assistive screening and tracking tools. None of these pathways replaces a physician or formal diagnostic testing. In case of acute chest discomfort or emergency symptoms, contact emergency services immediately.
        </p>
      </section>

      {/* Disease-Specific Screening Modules Section */}
      <div className="space-y-3 pt-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
          <div>
            <h2 className="text-base font-bold text-slate-900">
              Disease-Specific Clinical Screening Modules
            </h2>
            <p className="text-xs text-slate-500">
              When you have structured diagnostic test or laboratory reports, select an individual screening module below.
            </p>
          </div>
          <Badge variant="default" size="sm">Laboratory Data Required</Badge>
        </div>

        {/* ML Integration Status Notice */}
        <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-slate-700 text-xs space-y-1 leading-relaxed">
          <div className="flex items-center gap-2 font-bold text-slate-800">
            <Info className="w-4 h-4 text-teal-600 flex-shrink-0" />
            <span>AI-Assisted Screening Engine Active — Heart & Liver Disease</span>
          </div>
          <p className="text-slate-500">
            The <strong>Heart Disease</strong> screening module connects to the production Logistic Regression (Robust L1) model with configured screening threshold (t=0.31) and SHAP explainability. The <strong>Liver Disease</strong> module connects to the isotonic calibrated model. Both are orchestrated via Spring Boot and served by the FastAPI ML microservice. All stored records represent authentic clinical screening history.
          </p>
        </div>
      </div>

      {/* Disease Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {diseaseCards.map((card) => {
          const Icon = card.icon;
          const isHeart = card.type === 'HEART_DISEASE';
          return (
            <Card
              key={card.type}
              hoverable
              onClick={() => (isHeart ? setShowHeartPathwaysModal(true) : setSelectedDisease(card.type))}
              className="p-5 flex flex-col justify-between"
            >
              <div>
                <div className={`w-10 h-10 rounded-xl ${card.color} flex items-center justify-center mb-3 font-bold`}>
                  <Icon className="w-5 h-5" />
                </div>
                <h3 className="text-base font-bold text-slate-800">{card.title}</h3>
                <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">{card.description}</p>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs font-semibold text-teal-600">
                <span>{isHeart ? 'Explore Heart Pathways' : 'Configure Input'}</span>
                <ChevronRight className="w-4 h-4" />
              </div>
            </Card>
          );
        })}
      </div>

      {/* Assessment History Section */}
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-card space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
          <div>
            <h2 className="text-base font-bold text-slate-900">Stored ML Risk Assessments</h2>
            <p className="text-xs text-slate-400">
              Screening records saved for Patient #{activePatientId}
            </p>
          </div>

          {/* Filter Tabs */}
          <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-xl">
            <button
              onClick={() => setActiveHistoryTab('all')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                activeHistoryTab === 'all'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-500 hover:text-slate-800'
              }`}
            >
              All Reports ({assessments.length})
            </button>
            <button
              onClick={() => setActiveHistoryTab('heart')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-all ${
                activeHistoryTab === 'heart'
                  ? 'bg-rose-50 text-rose-700 shadow-xs'
                  : 'text-slate-500 hover:text-slate-800'
              }`}
            >
              <Heart className="w-3.5 h-3.5 text-rose-500" />
              <span>Heart History</span>
            </button>
            <button
              onClick={() => setActiveHistoryTab('liver')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-all ${
                activeHistoryTab === 'liver'
                  ? 'bg-amber-50 text-amber-700 shadow-xs'
                  : 'text-slate-500 hover:text-slate-800'
              }`}
            >
              <ShieldAlert className="w-3.5 h-3.5 text-amber-500" />
              <span>Liver History</span>
            </button>
          </div>
        </div>

        {/* Tab 1: Heart Disease Dedicated History */}
        {activeHistoryTab === 'heart' && (
          <HeartAssessmentHistory
            patientId={activePatientId}
            onSelectAssessment={(assessment) => setSelectedHeartAssessment(assessment)}
            onStartNewAssessment={() => setSelectedDisease('HEART_DISEASE')}
            refreshTrigger={refreshHistoryCount}
          />
        )}

        {/* Tab 2: Liver Disease History Filter */}
        {activeHistoryTab === 'liver' && (
          <div>
            {assessments.filter((a) => a.diseaseType === 'LIVER_DISEASE').length === 0 ? (
              <div className="py-10 text-center bg-slate-50/60 rounded-xl border border-dashed border-slate-200 p-6">
                <ShieldAlert className="w-10 h-10 text-slate-300 mx-auto mb-2" />
                <h4 className="text-sm font-semibold text-slate-700">No Liver Assessment Records</h4>
                <p className="text-xs text-slate-400 max-w-sm mx-auto mt-1">
                  No liver disease screening assessments found for this patient.
                </p>
              </div>
            ) : (
              <div className="divide-y divide-slate-100">
                {assessments
                  .filter((a) => a.diseaseType === 'LIVER_DISEASE')
                  .map((ass) => (
                    <div key={ass.id} className="py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div className="flex items-start gap-3">
                        <div className="w-9 h-9 rounded-xl bg-amber-50 text-amber-700 flex items-center justify-center font-bold text-xs flex-shrink-0 mt-0.5">
                          <ShieldAlert className="w-5 h-5" />
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-bold text-slate-900">{ass.diseaseType}</span>
                            {ass.riskLevel && (
                              <Badge
                                variant={
                                  ass.riskLevel === 'LOW'
                                    ? 'success'
                                    : ass.riskLevel === 'HIGH'
                                    ? 'danger'
                                    : 'warning'
                                }
                              >
                                {ass.riskLevel} Risk
                              </Badge>
                            )}
                          </div>
                          <span className="text-[11px] text-slate-400 flex items-center gap-1 mt-1">
                            <Clock className="w-3 h-3" />
                            {formatDateTime(ass.assessedAt)}
                          </span>
                        </div>
                      </div>

                      <div className="text-right sm:flex-shrink-0">
                        <span className="text-xs text-slate-400 block font-medium">Calibrated Risk Score</span>
                        <span className="text-lg font-bold text-slate-900">{ass.riskScore.toFixed(3)}</span>
                      </div>
                    </div>
                  ))}
              </div>
            )}
          </div>
        )}

        {/* Tab 3: All Assessments (Default List View) */}
        {activeHistoryTab === 'all' && (
          <div>
            {isLoading ? (
              <div className="py-8 text-center text-xs text-slate-400">
                Loading assessment history...
              </div>
            ) : assessments.length === 0 ? (
              <div className="py-10 text-center bg-slate-50/60 rounded-xl border border-dashed border-slate-200 p-6">
                <FileCheck2 className="w-10 h-10 text-slate-300 mx-auto mb-2" />
                <h4 className="text-sm font-semibold text-slate-700">No Assessment Records Found</h4>
                <p className="text-xs text-slate-400 max-w-sm mx-auto mt-1">
                  There are no stored disease assessments for this patient profile in the backend database.
                </p>
              </div>
            ) : (
              <div className="divide-y divide-slate-100">
                {assessments.map((ass) => {
                  const isHeartAss = ass.diseaseType === 'HEART_DISEASE';
                  return (
                    <div
                      key={ass.id}
                      onClick={() => {
                        if (isHeartAss) {
                          setSelectedHeartAssessment(ass);
                        }
                      }}
                      className={`py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 ${
                        isHeartAss
                          ? 'cursor-pointer hover:bg-rose-50/30 px-3 -mx-3 rounded-xl transition-colors'
                          : ''
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <div
                          className={`w-9 h-9 rounded-xl flex items-center justify-center font-bold text-xs flex-shrink-0 mt-0.5 ${
                            isHeartAss
                              ? 'bg-rose-50 text-rose-600'
                              : 'bg-teal-50 text-teal-700'
                          }`}
                        >
                          {isHeartAss ? <Heart className="w-5 h-5" /> : <FileCheck2 className="w-5 h-5" />}
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-bold text-slate-900">{ass.diseaseType}</span>
                            {ass.riskLevel ? (
                              <Badge
                                variant={
                                  ass.riskLevel === 'LOW'
                                    ? 'success'
                                    : ass.riskLevel === 'HIGH'
                                    ? 'danger'
                                    : 'warning'
                                }
                              >
                                {ass.riskLevel} Risk
                              </Badge>
                            ) : null}
                            {isHeartAss && (
                              <span className="text-[10px] font-semibold text-rose-600 bg-rose-50 px-2 py-0.5 rounded-full border border-rose-200">
                                Click to View Details
                              </span>
                            )}
                          </div>
                          <span className="text-[11px] text-slate-400 flex items-center gap-1 mt-1">
                            <Clock className="w-3 h-3" />
                            {formatDateTime(ass.assessedAt)}
                          </span>
                        </div>
                      </div>

                      <div className="flex items-center justify-between sm:justify-end gap-3 text-right sm:flex-shrink-0">
                        <div>
                          <span className="text-xs text-slate-400 block font-medium">
                            {isHeartAss ? 'Model Probability' : 'Risk Score'}
                          </span>
                          <span className="text-lg font-bold text-slate-900 font-mono">
                            {isHeartAss ? `${(ass.riskScore * 100).toFixed(1)}%` : ass.riskScore.toFixed(3)}
                          </span>
                        </div>
                        {isHeartAss && <ChevronRight className="w-4 h-4 text-rose-500" />}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Liver Disease Real Clinical Feature Input Modal */}
      <LiverDiseaseAssessmentModal
        isOpen={selectedDisease === 'LIVER_DISEASE'}
        onClose={() => setSelectedDisease(null)}
        onAssessmentComplete={handleAssessmentComplete}
        defaultAge={
          currentUser?.dateOfBirth
            ? Math.max(1, new Date().getFullYear() - new Date(currentUser.dateOfBirth).getFullYear())
            : null
        }
        defaultGender={currentUser?.gender}
      />

      {/* Heart Disease Real Clinical Feature Input Modal */}
      <HeartDiseaseAssessmentModal
        isOpen={selectedDisease === 'HEART_DISEASE'}
        onClose={() => setSelectedDisease(null)}
        onAssessmentComplete={handleAssessmentComplete}
        defaultAge={
          currentUser?.dateOfBirth
            ? Math.max(1, new Date().getFullYear() - new Date(currentUser.dateOfBirth).getFullYear())
            : null
        }
        defaultGender={currentUser?.gender}
      />

      {/* Heart Disease Detailed Assessment Modal (Stage 6) */}
      <HeartAssessmentDetailModal
        isOpen={selectedHeartAssessment !== null}
        onClose={() => setSelectedHeartAssessment(null)}
        assessment={selectedHeartAssessment}
      />

      {/* Generic Interface Ready Modal for other diseases (Diabetes, Parkinson's) */}
      {selectedDisease && selectedDisease !== 'LIVER_DISEASE' && selectedDisease !== 'HEART_DISEASE' && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-sm">
          <div className="bg-white rounded-2xl max-w-md w-full p-6 shadow-xl border border-slate-200 animate-in fade-in">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <h3 className="text-base font-bold text-slate-900">
                {diseaseCards.find((c) => c.type === selectedDisease)?.title || selectedDisease} Interface
              </h3>
              <button
                onClick={() => setSelectedDisease(null)}
                className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="py-6 text-center space-y-3">
              <div className="w-12 h-12 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center mx-auto">
                <Sparkles className="w-6 h-6" />
              </div>
              <h4 className="text-sm font-bold text-slate-800">
                Assessment Interface Ready
              </h4>
              <p className="text-xs text-slate-500 leading-relaxed max-w-sm mx-auto">
                Input fields and validation parameters for{' '}
                {diseaseCards.find((c) => c.type === selectedDisease)?.title || selectedDisease} are prepared. In
                accordance with clinical safety guidelines, live ML inference will be activated once the production model
                serving microservice is deployed in Phase 12.
              </p>
            </div>

            <div className="pt-3 border-t border-slate-100 flex justify-end">
              <Button variant="teal" size="sm" onClick={() => setSelectedDisease(null)}>
                Close
              </Button>
            </div>
          </div>
        </div>
      )}
      {/* Heart Health Pathways Modal */}
      {showHeartPathwaysModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm">
          <div className="bg-white rounded-2xl max-w-lg w-full p-6 shadow-2xl border border-slate-200 animate-in fade-in zoom-in-95 duration-150 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 rounded-xl bg-rose-50 text-rose-600 flex items-center justify-center font-bold">
                  <Heart className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900">Heart Health</h3>
                  <p className="text-xs text-slate-500">How can we help?</p>
                </div>
              </div>
              <button
                onClick={() => setShowHeartPathwaysModal(false)}
                className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3">
              {/* Option 1: Symptoms */}
              <div className="p-3.5 rounded-xl border border-slate-200 hover:border-teal-300 hover:bg-teal-50/20 transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="space-y-0.5">
                  <h4 className="text-xs font-bold text-slate-800">Talk about symptoms</h4>
                  <p className="text-[11px] text-slate-500 leading-relaxed">
                    Discuss what you're experiencing and get safety-focused conversational guidance.
                  </p>
                </div>
                <Button
                  variant="teal"
                  size="sm"
                  onClick={() => {
                    setShowHeartPathwaysModal(false);
                    navigate('/chat');
                  }}
                  className="flex-shrink-0 text-xs font-bold"
                  aria-label="Talk to Assistant about symptoms"
                >
                  Talk to Assistant
                </Button>
              </div>

              {/* Option 2: Health Monitoring */}
              <div className="p-3.5 rounded-xl border border-slate-200 hover:border-blue-300 hover:bg-blue-50/20 transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="space-y-0.5">
                  <h4 className="text-xs font-bold text-slate-800">Monitor my health</h4>
                  <p className="text-[11px] text-slate-500 leading-relaxed">
                    Track available health measurements, personal baselines, and changes over time.
                  </p>
                </div>
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => {
                    setShowHeartPathwaysModal(false);
                    navigate('/monitoring');
                  }}
                  className="flex-shrink-0 text-xs font-bold"
                  aria-label="Open Health Monitoring"
                >
                  Open Monitoring
                </Button>
              </div>

              {/* Option 3: Clinical Data Screening */}
              <div className="p-3.5 rounded-xl border border-rose-200 bg-rose-50/20 hover:bg-rose-50/40 transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="space-y-0.5">
                  <h4 className="text-xs font-bold text-slate-800">I have clinical results</h4>
                  <p className="text-[11px] text-slate-500 leading-relaxed">
                    Use validated clinical or laboratory measurements for the Heart screening assessment.
                  </p>
                  <p className="text-[10px] text-rose-600/80">
                    This option is for patients who already have the required validated clinical measurements.
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setShowHeartPathwaysModal(false);
                    setSelectedDisease('HEART_DISEASE');
                  }}
                  className="flex-shrink-0 text-xs font-bold border-rose-200 text-rose-700 hover:bg-rose-50"
                  aria-label="Open Clinical Assessment"
                >
                  Clinical Assessment
                </Button>
              </div>
            </div>

            <p className="text-[11px] text-slate-400 pt-1">
              * Non-diagnostic screening notice: None of these pathways replaces a doctor or diagnostic testing. In case of emergency distress, call 911 or visit an emergency room immediately.
            </p>
          </div>
        </div>
      )}

      {/* Disease Selection Gateway Modal ("I have clinical results") */}
      {showClinicalGatewayModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm">
          <div className="bg-white rounded-2xl max-w-xl w-full p-6 shadow-2xl border border-slate-200 animate-in fade-in zoom-in-95 duration-150 space-y-5">
            {/* Header */}
            <div className="flex items-start justify-between pb-3 border-b border-slate-100">
              <div>
                <h3 className="text-lg font-bold text-slate-900">
                  Which health assessment do you want to explore?
                </h3>
                <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                  Choose the screening module that matches the clinical or laboratory results you have available.
                </p>
              </div>
              <button
                onClick={() => setShowClinicalGatewayModal(false)}
                className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100 flex-shrink-0"
                aria-label="Close dialog"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Disease Modules List */}
            <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
              {/* 1. Heart Disease (Available) */}
              <div className="p-4 rounded-xl border border-rose-200/80 bg-rose-50/20 hover:bg-rose-50/40 transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 rounded-xl bg-rose-50 text-rose-600 flex items-center justify-center font-bold flex-shrink-0 mt-0.5 sm:mt-0">
                    <Heart className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h4 className="text-sm font-bold text-slate-900">Heart Disease</h4>
                      <Badge variant="teal" size="sm">Available Now</Badge>
                    </div>
                    <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">
                      Uses 13 clinical & laboratory measurements (resting BP, cholesterol, ECG, thalach, etc.).
                    </p>
                  </div>
                </div>
                <Button
                  variant="teal"
                  size="sm"
                  onClick={() => {
                    setShowClinicalGatewayModal(false);
                    setSelectedDisease('HEART_DISEASE');
                  }}
                  className="flex-shrink-0 text-xs font-bold"
                  aria-label="Open Heart Disease Assessment"
                >
                  Open Assessment
                </Button>
              </div>

              {/* 2. Liver Disease (Available) */}
              <div className="p-4 rounded-xl border border-amber-200/80 bg-amber-50/20 hover:bg-amber-50/40 transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center font-bold flex-shrink-0 mt-0.5 sm:mt-0">
                    <ShieldAlert className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h4 className="text-sm font-bold text-slate-900">Liver Disease</h4>
                      <Badge variant="teal" size="sm">Available Now</Badge>
                    </div>
                    <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">
                      Uses 10 hepatic biochemistry markers (bilirubin, enzymes, proteins, albumin, A/G ratio).
                    </p>
                  </div>
                </div>
                <Button
                  variant="teal"
                  size="sm"
                  onClick={() => {
                    setShowClinicalGatewayModal(false);
                    setSelectedDisease('LIVER_DISEASE');
                  }}
                  className="flex-shrink-0 text-xs font-bold"
                  aria-label="Open Liver Disease Assessment"
                >
                  Open Assessment
                </Button>
              </div>

              {/* 3. Diabetes (Coming Soon) */}
              <div className="p-4 rounded-xl border border-slate-200 bg-slate-50/60 opacity-80 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 rounded-xl bg-slate-100 text-slate-400 flex items-center justify-center font-bold flex-shrink-0 mt-0.5 sm:mt-0">
                    <Droplet className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h4 className="text-sm font-bold text-slate-700">Diabetes</h4>
                      <Badge variant="default" size="sm">Coming Soon</Badge>
                    </div>
                    <p className="text-xs text-slate-400 mt-0.5 leading-relaxed">
                      Metabolic screening module based on fasting glucose, insulin, and BMI markers.
                    </p>
                  </div>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  disabled
                  className="flex-shrink-0 text-xs font-semibold opacity-60 cursor-not-allowed bg-slate-100 border-slate-200 text-slate-400"
                  aria-label="Diabetes Assessment Coming Soon"
                >
                  Coming Soon
                </Button>
              </div>

              {/* 4. Parkinson's Disease (Coming Soon) */}
              <div className="p-4 rounded-xl border border-slate-200 bg-slate-50/60 opacity-80 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 rounded-xl bg-slate-100 text-slate-400 flex items-center justify-center font-bold flex-shrink-0 mt-0.5 sm:mt-0">
                    <Brain className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h4 className="text-sm font-bold text-slate-700">Parkinson's Disease</h4>
                      <Badge variant="default" size="sm">Coming Soon</Badge>
                    </div>
                    <p className="text-xs text-slate-400 mt-0.5 leading-relaxed">
                      Acoustic vocal frequency variations and motor biomarker characteristics.
                    </p>
                  </div>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  disabled
                  className="flex-shrink-0 text-xs font-semibold opacity-60 cursor-not-allowed bg-slate-100 border-slate-200 text-slate-400"
                  aria-label="Parkinson's Disease Assessment Coming Soon"
                >
                  Coming Soon
                </Button>
              </div>
            </div>

            {/* Footer with Cancel */}
            <div className="pt-3 border-t border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <p className="text-[11px] text-slate-400">
                * Non-diagnostic screening notice: Algorithmic screening tools require validated clinical measurements.
              </p>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowClinicalGatewayModal(false)}
                className="flex-shrink-0 text-xs font-semibold"
              >
                Cancel
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AssessmentsPage;
