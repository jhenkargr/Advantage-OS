import React, { useState, useEffect, useRef } from 'react';
import { 
  Palette, 
  Sparkles, 
  Image as ImageIcon, 
  Upload, 
  Download, 
  RefreshCw, 
  CheckCircle2
} from 'lucide-react';
import {
  generateTextToImage,
  generateImageToImage,
  generateCaption,
  fetchCreativeHealth,
  type GeneratedImageResult
} from '../services/api';
import type { CreativeHealth } from '../types/api';
import { ErrorState } from '../components/common/ErrorState';
import { Badge } from '../components/common/Badge';
import { SME_OPTIONS } from './TrendPredictorPage';

interface GeneratedAsset {
  id: string;
  type: 'text2img' | 'img2img';
  url: string;
  blob: Blob;
  prompt: string;
  modelUsed?: string;
  seedUsed?: string;
  timestamp: string;
}

interface CreativeStudioPageProps {
  initialPrompt?: string;
}

export const CreativeStudioPage: React.FC<CreativeStudioPageProps> = ({
  initialPrompt = '',
}) => {
  const [health, setHealth] = useState<CreativeHealth | null>(null);

  const [selectedSme, setSelectedSme] = useState<string>(SME_OPTIONS[0].backendKey);
  const [creativePrompt, setCreativePrompt] = useState<string>(initialPrompt || 'A premium product ad for a seasonal bakery offer with warm lighting and a luxury finish');
  const [captionText, setCaptionText] = useState<string>('Freshly baked daily with premium ingredients and a warm local touch.');
  const [negativePrompt, setNegativePrompt] = useState<string>('');
  const [imageSize, setImageSize] = useState<{ width: number; height: number }>({ width: 1024, height: 1024 });
  const [creativeSeed, setCreativeSeed] = useState<number>(-1);
  const [captionTone, setCaptionTone] = useState<'Playful' | 'Elegant' | 'Urgent'>('Playful');
  const [loadingCreative, setLoadingCreative] = useState<boolean>(false);
  const [loadingCaption, setLoadingCaption] = useState<boolean>(false);
  const [creativeResult, setCreativeResult] = useState<GeneratedImageResult | null>(null);
  const [creativeError, setCreativeError] = useState<string | null>(null);
  const [referenceImageFile, setReferenceImageFile] = useState<File | null>(null);
  const [referenceImagePreview, setReferenceImagePreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Session Gallery
  const [sessionGallery, setSessionGallery] = useState<GeneratedAsset[]>([]);

  // Update prompt if prop changes
  useEffect(() => {
    if (initialPrompt) {
      setCreativePrompt(initialPrompt);
    }
  }, [initialPrompt]);

  // Load Creative Health
  useEffect(() => {
    fetchCreativeHealth()
      .then(setHealth)
      .catch((err) => console.warn('Creative health check error:', err));
  }, []);

  const buildCreativeGenerationPrompt = () => {
    const smeLabel = SME_OPTIONS.find((option) => option.backendKey === selectedSme)?.label || 'SME';
    const normalizedCaption = captionText.trim();
    const normalizedPrompt = creativePrompt.trim();

    const requiredCaption = normalizedCaption || 'Your brand message';

    const instructionParts = [
      normalizedPrompt || 'Create an eye-catching brand campaign visual.',
      `For ${smeLabel} marketing.`,
      `CRITICAL REQUIREMENT: The exact text "${requiredCaption}" must appear visibly inside the final image as a branded overlay.`,
      'Do not omit, paraphrase, or move this text outside the image frame. It must be a visible, readable, in-scene text layer.',
      'Integrate the typography into the composition so it matches the design style, colors, lighting, and premium brand aesthetic of the visual.',
      'Place it as a clean, elegant, readable overlay that feels naturally part of the image, not as a separate sticker, watermark, or floating caption outside the scene.',
      'Use premium commercial photography styling, polished product presentation, studio-quality lighting, and a strong marketing campaign look.',
      'Keep the caption short, legible, and visually balanced within the composition, prioritizing clear readability over decorative clutter.',
    ];

    return instructionParts.join(' ');
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setReferenceImageFile(file);
      const url = URL.createObjectURL(file);
      setReferenceImagePreview(url);
    }
  };

  const handleGenerateCaption = async () => {
    if (!creativePrompt.trim()) {
      setCreativeError('Add a design brief first so the caption generator has something to write about.');
      return;
    }

    setLoadingCaption(true);
    setCreativeError(null);

    try {
      const response = await generateCaption(creativePrompt, captionTone);
      const generatedCaption = response?.caption?.trim();

      if (!generatedCaption) {
        throw new Error('Caption generator returned an empty caption.');
      }

      setCaptionText(generatedCaption);
    } catch (err: unknown) {
      console.error('Caption generation error:', err);
      setCreativeError(err instanceof Error ? err.message : 'Caption generation failed');
    } finally {
      setLoadingCaption(false);
    }
  };

  const addCaptionToImage = async (imageBlob: Blob, text: string): Promise<Blob> => {
    if (!text.trim()) return imageBlob;

    const image = await createImageBitmap(imageBlob);
    const canvas = document.createElement('canvas');
    const width = image.width;
    const height = image.height;

    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return imageBlob;

    ctx.drawImage(image, 0, 0, width, height);

    const fontSize = Math.max(28, Math.min(72, width * 0.06));
    const padding = Math.max(18, width * 0.03);
    const maxTextWidth = width * 0.7;
    const lineHeight = fontSize * 1.2;
    const words = text.split(/\s+/);
    const lines: string[] = [];
    let currentLine = '';

    for (const word of words) {
      const testLine = currentLine ? `${currentLine} ${word}` : word;
      ctx.font = `700 ${fontSize}px "Segoe UI", sans-serif`;
      if (ctx.measureText(testLine).width > maxTextWidth && currentLine) {
        lines.push(currentLine);
        currentLine = word;
      } else {
        currentLine = testLine;
      }
    }
    if (currentLine) lines.push(currentLine);

    const boxHeight = lines.length * lineHeight + padding * 2;
    const x = width / 2;
    const y = height * 0.8;
    const boxY = y - boxHeight / 2;

    ctx.fillStyle = 'rgba(17, 24, 39, 0.52)';
    ctx.fillRect(width * 0.12, boxY, width * 0.76, boxHeight);

    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#ffffff';
    ctx.font = `700 ${fontSize}px "Segoe UI", sans-serif`;
    ctx.shadowColor = 'rgba(0, 0, 0, 0.65)';
    ctx.shadowBlur = 12;

    lines.forEach((line, index) => {
      const lineY = boxY + padding + (index + 0.5) * lineHeight;
      ctx.fillText(line, x, lineY);
    });

    ctx.shadowBlur = 0;

    const finalBlob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob((blob) => resolve(blob), 'image/png');
    });

    return finalBlob || imageBlob;
  };

  const handleGenerateCreativeAsset = async () => {
    if (!captionText.trim() && !creativePrompt.trim()) return;
    setLoadingCreative(true);
    setCreativeError(null);

    try {
      const finalPrompt = buildCreativeGenerationPrompt();

      const result = referenceImageFile
        ? await generateImageToImage(finalPrompt, referenceImageFile, creativeSeed)
        : await generateTextToImage(finalPrompt, {
            width: imageSize.width,
            height: imageSize.height,
            seed: creativeSeed,
            negative_prompt: negativePrompt || undefined,
          });

      const finalBlob = captionText.trim() ? await addCaptionToImage(result.blob, captionText.trim()) : result.blob;
      const overlayUrl = URL.createObjectURL(finalBlob);
      const finalResult = {
        ...result,
        blob: finalBlob,
        objectUrl: overlayUrl,
      };

      setCreativeResult(finalResult);

      const newAsset: GeneratedAsset = {
        id: `${referenceImageFile ? 'i2i' : 't2i'}_${Date.now()}`,
        type: referenceImageFile ? 'img2img' : 'text2img',
        url: overlayUrl,
        blob: finalBlob,
        prompt: finalPrompt,
        modelUsed: result.modelUsed,
        seedUsed: result.seedUsed,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setSessionGallery((prev) => [newAsset, ...prev]);
    } catch (err: unknown) {
      console.error('Creative asset generation error:', err);
      setCreativeError(err instanceof Error ? err.message : 'Creative asset generation failed');
    } finally {
      setLoadingCreative(false);
    }
  };

  const handleDownload = (objectUrl: string, filename = 'advantage-creative.png') => {
    const a = document.createElement('a');
    a.href = objectUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="container" style={{ padding: '2.5rem 1.25rem 5rem 1.25rem' }}>
      {/* Page Title */}
      <div style={{ maxWidth: '780px', marginBottom: '2.5rem' }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', color: '#c084fc', marginBottom: '0.5rem', fontSize: '0.85rem', fontWeight: 600 }}>
          <Palette size={16} />
          <span>Generative AI Marketing Engine</span>
        </div>
        <h1 style={{ fontSize: '2.25rem', fontWeight: 800, marginBottom: '0.75rem', letterSpacing: '-0.025em' }}>
          Creative Studio
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.6 }}>
          Generate commercial-grade marketing materials powered by NVIDIA API Catalog (Flux &amp; Kontext), Groq LLM, and Pollinations.
        </p>
      </div>

      {/* Provider Status Pill */}
      {health && (
        <div className="glass-card" style={{ padding: '1rem 1.5rem', marginBottom: '2rem', display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <Sparkles size={18} color="#c084fc" />
            <div>
              <span style={{ fontSize: '0.86rem', fontWeight: 700, color: '#ffffff' }}>
                Active AI Studio Engine
              </span>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'block' }}>
                {health.providers}
              </span>
            </div>
          </div>
          <Badge variant="emerald" icon={<CheckCircle2 size={12} />}>
            API Gateway Ready
          </Badge>
        </div>
      )}

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))',
        gap: '2rem',
        alignItems: 'start',
      }}>
        <div className="glass-card" style={{ padding: '2rem' }}>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: '#ffffff', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Sparkles size={18} color="#c084fc" />
            Brand Creative Builder
          </h3>

          <div style={{ marginBottom: '1.25rem' }}>
            <label style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>
              SME Type
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '0.5rem' }}>
              {SME_OPTIONS.map((sme) => {
                const isSelected = selectedSme === sme.backendKey;
                return (
                  <button
                    key={sme.id}
                    type="button"
                    onClick={() => setSelectedSme(sme.backendKey)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      justifyContent: 'flex-start',
                      padding: '0.7rem 0.8rem',
                      borderRadius: 'var(--radius-sm)',
                      border: isSelected ? '1px solid var(--accent-violet)' : '1px solid var(--border-subtle)',
                      backgroundColor: isSelected ? 'rgba(139, 92, 246, 0.18)' : 'rgba(255, 255, 255, 0.02)',
                      color: isSelected ? '#ffffff' : 'var(--text-secondary)',
                      cursor: 'pointer',
                      textAlign: 'left',
                    }}
                  >
                    <span>{sme.icon}</span>
                    <span style={{ fontSize: '0.79rem', fontWeight: 600 }}>{sme.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          <div style={{ marginBottom: '1.25rem' }}>
            <label style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.4rem', fontWeight: 600 }}>
              Design Brief / Visual Prompt
            </label>
            <textarea
              rows={4}
              value={creativePrompt}
              onChange={(e) => setCreativePrompt(e.target.value)}
              placeholder="Describe your visual direction, product, campaign mood, and intended style..."
              className="textarea"
            />
          </div>

          <div style={{ marginBottom: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <label style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', display: 'block', fontWeight: 600 }}>
                Caption to print on the design
              </label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <select
                  value={captionTone}
                  onChange={(e) => setCaptionTone(e.target.value as 'Playful' | 'Elegant' | 'Urgent')}
                  className="select"
                  style={{ minWidth: '110px', height: '36px', fontSize: '0.75rem' }}
                >
                  <option value="Playful">Playful</option>
                  <option value="Elegant">Elegant</option>
                  <option value="Urgent">Urgent</option>
                </select>
                <button
                  type="button"
                  onClick={handleGenerateCaption}
                  disabled={loadingCaption || !creativePrompt.trim()}
                  className="btn btn-secondary"
                  style={{ fontSize: '0.75rem', padding: '0.45rem 0.7rem' }}
                >
                  {loadingCaption ? 'Generating...' : 'Generate Caption'}
                </button>
              </div>
            </div>
            <textarea
              rows={3}
              value={captionText}
              onChange={(e) => setCaptionText(e.target.value)}
              placeholder="Write the exact marketing caption message you want embedded inside the image design..."
              className="textarea"
            />
          </div>

          <div style={{ marginBottom: '1.25rem' }}>
            <label style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.4rem' }}>
              Negative Prompt (Optional)
            </label>
            <input
              type="text"
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              placeholder="blurry, distorted, low resolution, poor lighting..."
              className="input"
              style={{ fontSize: '0.85rem' }}
            />
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>
              Optional Reference Image
            </label>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              style={{ display: 'none' }}
            />

            {referenceImagePreview ? (
              <div style={{ position: 'relative', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border-card)', maxHeight: '180px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(0, 0, 0, 0.4)' }}>
                <img src={referenceImagePreview} alt="Reference preview" style={{ maxHeight: '180px', width: 'auto', objectFit: 'contain' }} />
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="btn btn-secondary"
                  style={{ position: 'absolute', bottom: '0.5rem', right: '0.5rem', fontSize: '0.75rem', padding: '0.35rem 0.65rem' }}
                >
                  Change Image
                </button>
              </div>
            ) : (
              <div
                onClick={() => fileInputRef.current?.click()}
                style={{ border: '2px dashed var(--border-card)', borderRadius: 'var(--radius-md)', padding: '2rem 1rem', textAlign: 'center', cursor: 'pointer', backgroundColor: 'rgba(255, 255, 255, 0.02)', transition: 'all 150ms ease' }}
                onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--accent-violet)'; e.currentTarget.style.backgroundColor = 'rgba(139, 92, 246, 0.05)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border-card)'; e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.02)'; }}
              >
                <Upload size={30} color="#c084fc" style={{ marginBottom: '0.75rem' }} />
                <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#ffffff' }}>Optional product photo</div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>If uploaded, the app uses image-to-image mode and prints the caption inside the final design.</span>
              </div>
            )}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.75rem' }}>
            <div>
              <label style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.4rem' }}>
                Aspect Ratio
              </label>
              <select
                value={`${imageSize.width}x${imageSize.height}`}
                onChange={(e) => {
                  const [w, h] = e.target.value.split('x').map(Number);
                  setImageSize({ width: w, height: h });
                }}
                className="select"
              >
                <option value="1024x1024">1:1 Square</option>
                <option value="768x1024">Portrait</option>
                <option value="1024x768">Landscape</option>
              </select>
            </div>
            <div>
              <label style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.4rem' }}>
                Seed (-1 = Random)
              </label>
              <input type="number" value={creativeSeed} onChange={(e) => setCreativeSeed(Number(e.target.value))} className="input" />
            </div>
          </div>

          <button
            type="button"
            onClick={handleGenerateCreativeAsset}
            disabled={loadingCreative || (!captionText.trim() && !creativePrompt.trim())}
            className="btn btn-primary"
            style={{ width: '100%', padding: '0.85rem', fontSize: '0.95rem', background: 'linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)' }}
          >
            <Sparkles size={18} />
            {loadingCreative ? 'Generating branded creative...' : referenceImageFile ? 'Generate with Image-to-Image' : 'Generate with Text-to-Image'}
          </button>

          {creativeError && (
            <div style={{ marginTop: '1.25rem' }}>
              <ErrorState message={creativeError} onRetry={handleGenerateCreativeAsset} />
            </div>
          )}
        </div>

        <div className="glass-card" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center', minHeight: '440px' }}>
          <h4 style={{ fontSize: '1rem', fontWeight: 700, color: '#ffffff', marginBottom: '1.25rem', alignSelf: 'flex-start' }}>
            Generated Creative Preview
          </h4>

          {loadingCreative && (
            <div style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', margin: 'auto' }}>
              <RefreshCw size={36} color="#c084fc" className="animate-spin" />
              <span style={{ color: '#a5b4fc', fontSize: '0.9rem' }}>
                {referenceImageFile ? 'Applying image-to-image transformation with the caption embedded in the design...' : 'Generating a new campaign image with the caption placed in the composition...'}
              </span>
              <div className="skeleton" style={{ width: '100%', maxWidth: '380px', height: '340px' }} />
            </div>
          )}

          {!loadingCreative && creativeResult && (
            <div className="animate-fade-in" style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <div style={{ position: 'relative', width: '100%', maxWidth: '460px', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border-card)', boxShadow: '0 8px 30px rgba(0, 0, 0, 0.6)', marginBottom: '1.25rem' }}>
                <img src={creativeResult.objectUrl} alt={buildCreativeGenerationPrompt()} style={{ width: '100%', height: 'auto', display: 'block' }} />
              </div>

              <div style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', paddingTop: '1rem', borderTop: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  <span>Mode: <strong style={{ color: '#ffffff' }}>{referenceImageFile ? 'Image-to-Image' : 'Text-to-Image'}</strong></span>
                  {creativeResult.seedUsed && (<span style={{ marginLeft: '0.75rem', fontFamily: 'var(--font-mono)' }}>Seed: {creativeResult.seedUsed}</span>)}
                </div>
                <button onClick={() => handleDownload(creativeResult.objectUrl, `${referenceImageFile ? 'img2img' : 'text2img'}-creative.png`)} className="btn btn-secondary" style={{ fontSize: '0.8rem', padding: '0.45rem 0.9rem' }}>
                  <Download size={14} />
                  Download PNG
                </button>
              </div>
            </div>
          )}

          {!loadingCreative && !creativeResult && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', margin: 'auto', color: 'var(--text-muted)', textAlign: 'center' }}>
              <ImageIcon size={48} strokeWidth={1.5} style={{ marginBottom: '1rem', opacity: 0.4 }} />
              <p style={{ fontSize: '0.9rem', maxWidth: '320px' }}>
                Select an SME, add the caption text, and optionally upload an image to generate a branded campaign design.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* SESSION CREATIVE GALLERY */}
      {sessionGallery.length > 0 && (
        <section style={{ marginTop: '4rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '1.25rem' }}>
            <div>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#ffffff' }}>
                Session Creative Gallery
              </h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Visuals generated during this active dashboard session
              </p>
            </div>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              {sessionGallery.length} Saved Assets
            </span>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
            gap: '1.25rem',
          }}>
            {sessionGallery.map((asset) => (
              <div
                key={asset.id}
                className="glass-card glass-card-hover"
                style={{ padding: '0.75rem', display: 'flex', flexDirection: 'column' }}
              >
                <div style={{
                  position: 'relative',
                  width: '100%',
                  aspectRatio: '1',
                  borderRadius: 'var(--radius-sm)',
                  overflow: 'hidden',
                  marginBottom: '0.75rem',
                  backgroundColor: '#000000',
                }}>
                  <img
                    src={asset.url}
                    alt={asset.prompt}
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden', marginBottom: '0.5rem', flex: 1 }}>
                  {asset.prompt}
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{asset.timestamp}</span>
                  <button
                    onClick={() => handleDownload(asset.url, `creative-${asset.id}.png`)}
                    className="btn btn-secondary"
                    style={{ padding: '0.3rem 0.6rem', fontSize: '0.72rem' }}
                  >
                    <Download size={12} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
};
