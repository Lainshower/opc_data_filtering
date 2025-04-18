import oop.ex2.GameGUI;
import java.awt.*;

/**
 * Represents a human-controlled spaceship.
 * @author jherskow
 */
public class HumanShip extends SpaceShip {

	/**
	 * Implements user control of the ship.
	 * @param Game The game object.
	 */
	public void doAction(SpaceWars Game){
		boolean isThrottle = false;
		int turn = 0;
		if (Game.getGUI().isTeleportPressed()){
			teleport();
		}
		if(Game.getGUI().isRightPressed()) turn--;
		if(Game.getGUI().isLeftPressed()) turn++;
		if (Game.getGUI().isUpPressed()) isThrottle = true;
		getPhysics().move(isThrottle,turn);
		if(Game.getGUI().isShieldsPressed()) shieldOn();
		else shieldsUp = false;
		if(Game.getGUI().isShotPressed()) fire(Game);
		rechargeEnergy();
	}

	/** This function overrides the default image to differentiate the human player's
	 * ship from the others.
	 * @return The image sprite for the ship.
	 */
	@Override
	public Image getImage(){return shieldsUp ? GameGUI.SPACESHIP_IMAGE_SHIELD : GameGUI.SPACESHIP_IMAGE;}
}
