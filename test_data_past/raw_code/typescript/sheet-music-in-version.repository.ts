import { SheetMusicInVersionProjection } from '../projection/sheet-music-in-version/sheet-music-in-version.projection';

export abstract class SheetMusicInVersionRepository {
  abstract save(
    projection: SheetMusicInVersionProjection,
    type: string,
    occurredAt: Date,
  ): Promise<void>;

  abstract getPrevious(): Promise<SheetMusicInVersionProjection>;
}
