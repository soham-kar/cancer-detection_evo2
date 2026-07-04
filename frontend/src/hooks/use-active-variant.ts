import { useActiveVariantContext } from "~/contexts/active-variant";

export function useActiveVariant() {
  return useActiveVariantContext();
}
