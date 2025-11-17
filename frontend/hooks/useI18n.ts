"use client";
import { useI18nCtx } from "../providers/I18nProvider";
export function useI18n() {
  return useI18nCtx();
}
