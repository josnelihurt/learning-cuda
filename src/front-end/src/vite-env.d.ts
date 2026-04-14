/// <reference types="vite/client" />
/// <reference types="react" />

declare const __APP_VERSION__: string;
declare const __APP_BRANCH__: string;
declare const __BUILD_TIME__: string;

declare global {
  namespace JSX {
    interface IntrinsicElements {
      'information-banner': React.DetailedHTMLProps<
        React.HTMLAttributes<HTMLElement>,
        HTMLElement
      >;
    }
  }
}

export {};
