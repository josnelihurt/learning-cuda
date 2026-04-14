export function InformationBanner() {
  return (
    <div
      style={{
        background: 'linear-gradient(90deg, #ff6b35 0%, #f7931e 100%)',
        color: 'white',
        padding: '4px 0',
        overflow: 'hidden',
        whiteSpace: 'nowrap',
        fontSize: '13px',
        fontWeight: 500,
        borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
      }}
    >
      <div style={{ display: 'inline-block', paddingLeft: '100%', animation: 'react-banner-marquee 25s linear infinite' }}>
        Production deployment in progress - some components may be unavailable
      </div>
    </div>
  );
}
