namespace CoreTrainer.Graphics.UI;

/// <summary>
/// A UI control that holds and manages child controls, forming the visual tree.
/// Children inherit their theme from the container unless they set one explicitly.
/// </summary>
public abstract class UIContainer : UIControl
{
	#region Fields

	private readonly List<UIControl> _children = [];

	#endregion

	#region Constructors

	protected UIContainer(int x, int y, int width, int height) : base(x, y, width, height)
	{
	}

	#endregion

	#region Properties

	/// <summary>
	/// Read-only view of this container's children, in insertion order.
	/// </summary>
	public IReadOnlyList<UIControl> Children => _children;

	#endregion

	#region Public Methods

	/// <summary>
	/// Adds a child control, linking it into the visual tree so it inherits this
	/// container's theme.
	/// </summary>
	/// <param name="child">Control to add.</param>
	/// <returns>The added control.</returns>
	public T Add<T>(T child) where T : UIControl
	{
		child.Parent = this;
		_children.Add(child);
		OnChildAdded(child);
		return child;
	}

	/// <summary>
	/// Removes a child control and detaches it from the visual tree.
	/// </summary>
	/// <param name="child">Control to remove.</param>
	/// <returns>True if the control was removed.</returns>
	public bool Remove(UIControl child)
	{
		var removed = _children.Remove(child);
		if (removed)
		{
			child.Parent = null;
			OnChildrenChanged();
		}
		return removed;
	}

	/// <summary>
	/// Removes all child controls, detaching each from the visual tree.
	/// </summary>
	public void Clear()
	{
		foreach (var child in _children)
		{
			child.Parent = null;
		}
		_children.Clear();
		OnChildrenChanged();
	}

	/// <summary>
	/// Updates all visible children.
	/// </summary>
	public override void Update()
	{
		if (!Visible) return;

		foreach (var child in _children)
		{
			child.Update();
		}
	}

	/// <summary>
	/// Renders all visible children. Derived containers that draw their own chrome
	/// (background, border) should override, draw, then call <c>base.Render()</c>.
	/// </summary>
	public override void Render()
	{
		if (!Visible) return;

		foreach (var child in _children)
		{
			child.Render();
		}
	}

	#endregion

	#region Protected Methods

	/// <summary>
	/// Called after a child is added so layout-managing containers can position it.
	/// Default implementation does nothing.
	/// </summary>
	/// <param name="child">The child that was just added.</param>
	protected virtual void OnChildAdded(UIControl child)
	{
		// Base implementation does nothing
	}

	/// <summary>
	/// Called after the child collection changes (removal or clear) so layout-managing
	/// containers can reflow. Default implementation does nothing.
	/// </summary>
	protected virtual void OnChildrenChanged()
	{
		// Base implementation does nothing
	}

	#endregion
}
